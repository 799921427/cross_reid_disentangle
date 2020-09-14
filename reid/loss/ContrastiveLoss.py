import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

    euclidean_distance = F.pairwise_distance(output1, output2)

    x1 = 1+(-1)*label
    x2 = torch.pow(euclidean_distance, 2)
	
    loss_contrastive = x1*x2
    x3 = label
    x4 = torch.pow(torch.clamp(self.margin + (-1.0)*euclidean_distance, min=0.0), 2)
    loss_contrastive += x3*x4
    loss_contrastive = torch.mean(loss_contrastive)

    return loss_contrastive
