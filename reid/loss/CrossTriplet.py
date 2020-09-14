from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class CrossTriplet(nn.Module):
    def __init__(self, margin=0):
        super(CrossTriplet, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, sub):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        sub = sub.expand(n, n).eq(sub.expand(n, n).t())
        sub = 1 - sub

        mask1 = mask * sub
        mask2 = (1-mask) * sub
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask1[i]].max())
            dist_an.append(dist[i][mask2[i]].min())
        
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        #print(dist_an,dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = dist_an.data > dist_ap.data
        return loss, prec
