from __future__ import print_function, absolute_import
import time
import random
import copy

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from reid.loss.CrossTriplet import CrossTriplet
from torch import nn
from .utils.meters import AverageMeter
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def to_gray(x): #simple
    x = torch.mean(x, dim=1, keepdim=True)
    return x

def scale2(x):
    if x.size(2) > 128: # do not need to scale the input
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')  #bicubic is not available for the time being.
    return x

class BaseTrainer(object):
    def __init__(self, gen_a, gen_b, dis_a, dis_b, criterion_z, criterion_I, criterion_att, trainvallabel, a, b, c, u, k):
        super(BaseTrainer, self).__init__()
        # self.id_a = id_a
        # self.id_b = id_b
        self.gen_a = gen_a
        self.gen_b = gen_b
        self.dis_a = dis_a
        self.dis_b = dis_b
        self.criterion_z = criterion_z
        self.criterion_I = criterion_I
        self.criterion_att = criterion_att
        self.trainvallabel = trainvallabel
        self.recon_s_w = 0
        self.recon_f_w = 0
        self.recon_x_w = 5
        self.recon_x_cyc_w = 0
        self.vgg_w = 0
        self.max_w = 1
        self.a = a
        self.b = b
        self.c = c
        self.u = u
        self.k = k
        self.batch_size = 8
        self.style_dim = 1
        self.s_a = torch.randn(self.batch_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(self.batch_size, self.style_dim, 1, 1).cuda()
        
    def train(self, epoch, data_loader, gen_opt, dis_opt, print_freq=1):
        # self.id_a.train()
        # self.id_b.train()
        self.gen_a.train()
        self.gen_b.train()
        self.dis_a.train()
        self.dis_b.train()
        self.dis_opt = dis_opt
        self.gen_opt = gen_opt
        # self.id_opt = id_opt
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses_generator = AverageMeter()
        # losses_triple = AverageMeter()
        # losses_idloss = AverageMeter()
        # losses_attention = AverageMeter()

        for i, inputs in enumerate(data_loader):
            # data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)
            # for ix, input in enumerate(inputs):
            #     img = input.cpu().clone()
            #     img = img.squeeze(0)
            #     unloader = transforms.ToPILImage()
            #     img = unloader(img)
            #     plt.imshow(img)
            #     plt.title(str(label[ix]))
            #     plt.pause(5)

            # self._forward(inputs, label, sub)
            a, b, x_ab, x_ba, s_a, s_b, c_a, c_b, x_a_recon, x_b_recon, f_a, f_b, p_a, p_b, labels = self._forward(inputs, label, sub)
            self.dis_update(x_ab.clone(), x_ba.clone(), a, b, num_gpu=1)
            self.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, x_a_recon, x_b_recon, a, b, c_a, c_b, p_a, p_b, labels, epoch, num_gpu=1)

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:])+1e-8))

    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:]**2)

    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])


    def dis_update(self, x_ab, x_ba, x_a, x_b, num_gpu):
        self.dis_opt.zero_grad()
        # D loss
        if num_gpu>1:
            self.loss_dis_a, reg_a = self.dis_a.module.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.module.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        else:
            self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
            self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = self.loss_dis_a + self.loss_dis_b
        print("DLoss: %.4f"%self.loss_dis_total )
        # if self.fp16:
        #     with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        self.loss_dis_total.backward()
        self.dis_opt.step()


    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, x_a_recon, x_b_recon, x_a, x_b, c_a, c_b, p_a, p_b, l, iteration, num_gpu):
        # ppa, ppb is the same person
        self.gen_opt.zero_grad()
        # self.id_opt.zero_grad()

        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0, 1)
        #################################
        # encode style
        if 0.5 >= rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen_b.enc_style(x_ab_copy)
            s_b_recon = self.gen_a.enc_style(x_ba_copy)
        else:
            # copy the encoder
            self.enc_content_copy_a = copy.deepcopy(self.gen_a.enc_style)
            self.enc_content_copy_a = self.enc_content_copy_a.eval()
            self.enc_content_copy_b = copy.deepcopy(self.gen_b.enc_style)
            self.enc_content_copy_b = self.enc_content_copy_b.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy_a(x_ab)
            s_b_recon = self.enc_content_copy_b(x_ba)

        #################################
        # encode appearance
        self.id_a_copy = copy.deepcopy(self.gen_a.enc_content)
        self.id_a_copy = self.id_a_copy.eval()
        self.id_b_copy = copy.deepcopy(self.gen_b.enc_content)
        self.id_b_copy = self.id_b_copy.eval()
        self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_b_copy.apply(train_bn)
        # encode again (encoder is fixed, input is tuned)
        content, f_a_recon, p_a_recon = self.id_a_copy(scale2(x_ba))
        content, f_b_recon, p_b_recon = self.id_b_copy(scale2(x_ab))

        # id loss
        self.loss_id = self.criterion_z(p_a, l) + self.criterion_z(p_b, l)
        self.loss_gen_recon_id = self.criterion_z(p_a_recon, l) + self.criterion_z(p_b_recon, l)
        # auto-encoder image reconstruction
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        # feature reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if self.recon_s_w > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if self.recon_s_w > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if self.recon_f_w > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if self.recon_f_w > 0 else 0

        x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if self.recon_x_cyc_w > 0 else None
        x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if self.recon_x_cyc_w > 0 else None


        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if self.recon_x_cyc_w  > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if self.recon_x_cyc_w  > 0 else 0
        # GAN loss
        if num_gpu > 1:
            self.loss_gen_adv_a = self.dis_a.module.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.module.calc_gen_loss(self.dis_b, x_ab)
        else:
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if self.vgg_w > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if self.vgg_w > 0 else 0

        if iteration > 60:
            self.recon_f_w += 0.01
            self.recon_f_w = min(self.recon_f_w, self.max_w)
            self.recon_s_w += 0.01
            self.recon_s_w = min(self.recon_s_w, self.max_w)
            self.recon_x_cyc_w += 0.01
            self.recon_x_cyc_w = min(self.recon_x_cyc_w, 2)

        # total loss
        self.loss_gen_total = self.loss_gen_adv_a + \
                              self.loss_gen_adv_b + \
                              self.recon_x_w * self.loss_gen_recon_x_a + \
                              self.recon_f_w * self.loss_gen_recon_f_a + \
                              self.recon_s_w * self.loss_gen_recon_s_a + \
                              self.recon_x_w * self.loss_gen_recon_x_b + \
                              self.recon_f_w * self.loss_gen_recon_f_b + \
                              self.recon_s_w * self.loss_gen_recon_s_b + \
                              self.recon_x_cyc_w * self.loss_gen_cycrecon_x_a + \
                              self.recon_x_cyc_w * self.loss_gen_cycrecon_x_b + \
                              self.loss_id + \
                              0.5 * self.loss_gen_recon_id
                              # self.vgg_w * self.loss_gen_vgg_a + \
                              # self.vgg_w * self.loss_gen_vgg_b

        self.loss_gen_total.backward()
        self.gen_opt.step()
        # self.id_opt.step()
        print("L_total: %.4f, L_gan: %.4f,  Lx: %.4f,  Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f" % (
            self.loss_gen_total, \
            (self.loss_gen_adv_a + self.loss_gen_adv_b), \
            self.recon_x_w * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
            self.recon_x_cyc_w * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
            self.recon_f_w * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
            self.recon_s_w * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
            0.5 * self.loss_gen_recon_id, \
            self.loss_id))


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, cams = inputs
        # for ix, input in enumerate(imgs):
        #     img = input.cpu().clone()
        #     img = img.squeeze(0)
        #     unloader = transforms.ToPILImage()
        #     img = unloader(img)
        #     plt.imshow(img)
        #     plt.title(fnames[ix])
        #     plt.pause(5)
        inputs = imgs.cuda()
        pids = pids.cuda()
        sub = ((cams == 2).long() + (cams == 5).long()).cuda()
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        return inputs, sub, label

    def _forward(self, inputs, label, sub):
        # print("hhhhhhhhhhhh")
        n = inputs.size(0)
        for i, t in enumerate(sub):
            if i%2 == 1: sub[i] = 1
        inputs_a = inputs[0:n:2,:,:,:]
        inputs_b = inputs[1:n:2,:,:,:]
        a_label = label[::2]

        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        # f_a, p_a = self.id_a(inputs_a)
        # f_b, p_b = self.id_b(inputs_b)
        # gray_inputs_a = to_gray(inputs_a)
        # gray_inputs_b = to_gray(inputs_b)
        c_a, s_a_fake, f_a, p_a = self.gen_a.encode(inputs_a)
        c_b, s_b_fake, f_b, p_b = self.gen_b.encode(inputs_b)
        # print("h1:",s_a_fake.size())
        # print("h2:", s_a.size())
        # # decode
        # print("input_size:", inputs_a.size())
        # print("style_size:", s_b.size())
        # print("id_size:", c_a.size())
        x_ba = self.gen_a.decode(c_b, s_a)
        # print("x_ba size", x_ba.size())
        x_ab = self.gen_b.decode(c_a, s_b)
        x_a_recon = self.gen_a.decode(c_a, s_a)
        x_b_recon = self.gen_b.decode(c_b, s_b)


        return inputs_a, inputs_b, x_ab, x_ba, s_a, s_b, c_a, c_b, x_a_recon, x_b_recon, f_a, f_b, p_a, p_b, a_label

