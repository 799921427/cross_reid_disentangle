from __future__ import print_function, absolute_import
import time
import random
from .models import networks
import torch
import numpy
from torch.autograd import Variable
from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from reid.loss.CrossTriplet import CrossTriplet
from torch import nn
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model_s, model_ir, model_t, criterion_z, criterion_z_s, criterion_I, criterion_att, trainvallabel, a, b, c, u, k):
        super(BaseTrainer, self).__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.model_ir = model_ir
        self.criterion_z = criterion_z
        self.criterion_z_s = criterion_z_s
        self.criterion_I = criterion_I
        self.criterion_att = criterion_att
        self.trainvallabel = trainvallabel
        # self.netG_A = networks.define_G(3,3,64,'resnet_9blocks','instance','store_false','normal',0.02,'0')
        # self.netG_B = networks.define_G(3,3,64,'resnet_9blocks','instance','store_false','normal',0.02,'0')
        self.a = a
        self.b = b
        self.c = c
        self.u = u
        self.k = k
        
    def train(self, epoch, data_loader, optimizer_generator_I, print_freq=1):
        self.model_t.train()
        self.model_s.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_generator = AverageMeter()
        losses_triple = AverageMeter()
        losses_idloss = AverageMeter()
        losses_idloss_s = AverageMeter()
        losses_idloss_ir = AverageMeter()
        losses_attention_s = AverageMeter()
        losses_attention_ir = AverageMeter()
        
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)

            # Calc the loss
            loss_t, loss_id, loss_id_s, loss_id_ir, loss_attention_s, loss_attention_ir = self._forward(inputs, label, sub)
            L = self.a * loss_t + self.b * loss_id + loss_id_s + loss_id_ir + 1.0 * loss_attention_s + 0.9 * loss_attention_ir
            # L = loss_attention

            neg_L = - self.u * L

            # if ((epoch * len(data_loader) + i) % self.k == 0):
            #     optimizer_discriminator.zero_grad()
            #     neg_L.backward()
            #     optimizer_discriminator.step()
            # else:
            optimizer_generator_I.zero_grad()
            L.backward()
            optimizer_generator_I.step()

            losses_generator.update(L.data.item(), label.size(0))
            losses_idloss.update(loss_id.item(), label.size(0))
            losses_idloss_s.update(loss_id_s.item(), label.size(0))
            losses_idloss_ir.update(loss_id_ir.item(), label.size(0))

            losses_triple.update(loss_t.item(), label.size(0))
            losses_attention_s.update(loss_attention_s.item(), label.size(0))
            losses_attention_ir.update(loss_attention_ir.item(), label.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Total Loss {:.3f} ({:.3f})\t'
                      'IDE Loss {:.3f} ({:.3f})\t'
                      'IDE Loss S {:.3f} ({:.3f})\t'
                      'IDE Loss IR {:.3f} ({:.3f})\t'
                      'Triple Loss {:.3f} ({:.3f})\t'
                      'Att Loss S {:.9f} ({:.9f})\t'
                      'Att Loss IR {:.9f} ({:.9f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_generator.val, losses_generator.avg,
                              losses_idloss.val, losses_idloss.avg,
                              losses_idloss_s.val, losses_idloss_s.avg,
                              losses_idloss_ir.val, losses_idloss_ir.avg,
                              losses_triple.val, losses_triple.avg,
                              losses_attention_s.val, losses_attention_s.avg,
                              losses_attention_ir.val, losses_attention_ir.avg))
        return losses_triple.avg, losses_generator.avg, losses_attention_s.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, cams = inputs
        # print(pids),
        # print(cams)
        inputs = imgs.cuda()
        pids = pids.cuda()
        sub = ((cams == 2).long() + (cams == 5).long()).cuda()
        # print(sub)
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        # print(label)
        return inputs, sub, label

    def _forward(self, inputs, label, sub):
        # print(inputs.size(0))
        # print(label)
        n = inputs.size(0)
        rgb_inputs = inputs[0:n:2,:,:,:]
        ir_inputs = inputs[1:n:2, :, :, :]
        att_label = label[::2]
        outputs, outputs_pool, att_feats, att_cls = self.model_t(inputs)

        loss_t, prec = self.criterion_I(outputs_pool, label, sub)
        #
        loss_id = self.criterion_z(outputs, label)
    #loss_id = loss_id + 0.1 * self.criterion_z(att_cls, label)

        outputs_ir, outputs_pool_ir, att_feats_ir, _ = self.model_ir(ir_inputs)
        outputs_s, outputs_pool_s, att_feats_s, _ = self.model_s(rgb_inputs)

        loss_id_s = self.criterion_z(outputs_s, att_label)
        loss_id_ir = self.criterion_z(outputs_ir, att_label)
        #
        loss_att_s = 0
        loss_att_ir = 0
        for i, fea in enumerate(att_feats):
            # print(fea)
            fea_s = fea[0:n:2, :]
            fea_ir = fea[1:n:2, :]
            fea_s = torch.nn.functional.normalize(fea_s, dim=1, p=2)
            att_feat_s = torch.nn.functional.normalize(att_feats_s[i], dim=1, p=2)
            loss_a = self.criterion_att(fea_s, att_feat_s.detach())
            # print(loss_a)
            loss_att_s += loss_a

            fea_ir = torch.nn.functional.normalize(fea_ir, dim=1, p=2)
            att_feat_ir = torch.nn.functional.normalize(att_feats_ir[i], dim=1, p=2)
            loss_a = self.criterion_att(fea_ir, att_feat_ir.detach())
            # print(loss_a)
            loss_att_ir += loss_a
        # # loss_att += self.criterion_att(g_att, g_att_s.detach())
        #
        # # outputs_discriminator = self.model_discriminator(outputs_pool)
        #
        # # loss_discriminator = self.criterion_D(outputs_discriminator, sub)
        #
        # loss_att = loss_att * 10
        # loss_id = 0.5 * loss_id
        return  loss_t, loss_id, loss_id_s, loss_id_ir, loss_att_s, loss_att_ir

