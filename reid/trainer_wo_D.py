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
    def __init__(self, model_generator_I, criterion_z, trainvallabel):
        super(BaseTrainer, self).__init__()
        self.model_generator_I = model_generator_I
        self.criterion_z = criterion_z
        self.trainvallabel = trainvallabel
        
    def train(self, epoch, data_loader, optimizer_generator_I, print_freq=1):
        self.model_generator_I.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_idloss = AverageMeter()
        
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, label = self._parse_data(inputs)
            # print(inputs)
            # print(inputs.size())
            # print(label)
            # print('hhh')
            # Calc the loss
            loss_id = self._forward(inputs, label)

            optimizer_generator_I.zero_grad()
            loss_id.backward()
            optimizer_generator_I.step()

            losses_idloss.update(loss_id.item(), label.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'IDE Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_idloss.val, losses_idloss.avg))
        return  losses_idloss.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, cams = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        return inputs, label

    def _forward(self, inputs, label):
        outputs, outputs_pool, _, att_out = self.model_generator_I(inputs)
        # print(outputs)
        # print(outputs_pool)
        # print(label)
        # print('hhh')
        # print(outputs.size())
        loss_id = self.criterion_z(outputs, label)
        loss_id = loss_id + self.criterion_z(att_out, label)
        # print(loss_id)
        return  loss_id
