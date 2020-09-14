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
    def __init__(self, model_t, criterion_z, criterion_I, criterion_att, trainvallabel, a, b, c, u, k):
        super(BaseTrainer, self).__init__()
        self.model_t = model_t
        self.criterion_z = criterion_z
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

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_generator = AverageMeter()
        losses_triple = AverageMeter()
        losses_idloss = AverageMeter()
        
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)

            # Calc the loss
            loss_t, loss_id = self._forward(inputs, label, sub)
            L = self.a * loss_t + self.b * loss_id
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
            losses_triple.update(loss_t.item(), label.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Total Loss {:.3f} ({:.3f})\t'
                      'IDE Loss {:.3f} ({:.3f})\t'
                      'Triple Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_generator.val, losses_generator.avg,
                              losses_idloss.val, losses_idloss.avg,
                              losses_triple.val, losses_triple.avg))
        return losses_triple.avg, losses_generator.avg

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
        n = inputs.size(0)

        # print(att_inputs.size())
        outputs, outputs_pool, att_feats, att_cls = self.model_t(inputs)

        loss_t, prec = self.criterion_I(outputs_pool, label, sub)
        #
        loss_id = self.criterion_z(outputs, label)

        # outputs, outputs_pool, att_feats, g_att = self.model_t(inputs)
        # outputs_s, outputs_pool_s, att_feats_s, g_att_s = self.model_s(inputs)

        #
        # loss_att = 0
        # for i, fea in enumerate(att_feats):
        #     # print(fea)
        #     fea = torch.nn.functional.normalize(fea, dim=1, p=2)
        #     # print(fea)
        #     att_feat = torch.nn.functional.normalize(att_feats_s[i], dim=1, p=2)
        #     loss_a = self.criterion_att(fea, att_feat.detach())
        #     # print(loss_a)
        #     loss_att += loss_a
        # # loss_att += self.criterion_att(g_att, g_att_s.detach())
        #
        # # outputs_discriminator = self.model_discriminator(outputs_pool)
        #
        # # loss_discriminator = self.criterion_D(outputs_discriminator, sub)
        #
        # loss_att = loss_att * 10
        return  loss_t, loss_id

