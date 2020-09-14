from __future__ import print_function, absolute_import
import time
import random

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from reid.loss.CrossTriplet import CrossTriplet
from torch import nn
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model_generator_I, model_discriminator, criterion_z, criterion_I, criterion_D, trainvallabel, a, b, c, u, k):
        # type: (object, object, object, object, object, object, object, object, object, object, object) -> object
        super(BaseTrainer, self).__init__()
        self.model_generator_I = model_generator_I
        self.model_discriminator = model_discriminator
        self.criterion_z = criterion_z
        self.criterion_I = criterion_I
        self.criterion_D = criterion_D
        self.trainvallabel = trainvallabel
        self.a = a
        self.b = b
        self.c = c
        self.u = u
        self.k = k
        
    def train(self, epoch, data_loader, optimizer_generator_I, optimizer_discriminator, print_freq=1):
        self.model_generator_I.train()
        self.model_discriminator.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_generator = AverageMeter()
        losses_triple = AverageMeter()
        losses_idloss = AverageMeter()
        losses_discriminator = AverageMeter()
        
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)

            # Calc the loss
            loss_t, loss_id, loss_discriminator = self._forward(inputs, label, sub)
            L = self.a * loss_t + self.b * loss_id - self.c * loss_discriminator

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
            losses_discriminator.update(loss_discriminator.item(), label.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Total Loss {:.3f} ({:.3f})\t'
                      'IDE Loss {:.3f} ({:.3f})\t'
                      'Triple Loss {:.3f} ({:.3f})\t'
                      'D Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_generator.val, losses_generator.avg,
                              losses_idloss.val, losses_idloss.avg,
                              losses_triple.val, losses_triple.avg,
                              losses_discriminator.val, losses_discriminator.avg))
        return losses_triple.avg, losses_generator.avg

    def _parse_data(self, inputs):
        # type: (object) -> object
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, cams = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        sub = ((cams == 2).long() + (cams == 5).long()).cuda()
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        return inputs, sub, label

    def _forward(self, inputs, label, sub):
        outputs, outputs_pool, _, _ = self.model_generator_I(inputs)

        loss_t ,prec = self.criterion_I(outputs_pool, label, sub)

        loss_id = self.criterion_z(outputs, label)

        outputs_discriminator = self.model_discriminator(outputs_pool)

        loss_discriminator = self.criterion_D(outputs_discriminator, sub)

        return  loss_t, loss_id, loss_discriminator
