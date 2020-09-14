from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['discriminator']


class discriminator(nn.Module):
    def __init__(self, input_dim=1024, output_dim=2):
        super(discriminator, self).__init__()

        self.FC1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.re1 = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)
        self.FC2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.re2 = nn.LeakyReLU(0.1)
        self.FC3 = nn.Linear(512, 2)

        self.reset_params()

    def forward(self, x):

        x = self.FC1(x)
        x = self.bn1(x)
        x = self.re1(x)
        x = self.drop(x)
        x = self.FC2(x)
        x = self.bn2(x)
        x = self.re2(x)
        x = self.FC3(x)

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class classifier(nn.Module):
    def __init__(self, input_dim=2048, num_features=0, norm=False, dropout=0, num_classes=0):
        super(classifier, self).__init__()

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        self.base = torchvision.models.resnet50(pretrained=True)
        out_planes = self.base.fc.in_features

        self.base = self.base.layer4

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):

        x = self.base(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.dropout > 0:
            x = self.drop(x)
        feature = F.normalize(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x, feature


def Discriminator(**kwargs):
    return discriminator(**kwargs)


def Classifier(**kwargs):
    return classifier(**kwargs)