from __future__ import absolute_import

import math
import torch
from torch.nn import Parameter
from torch import cat
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152','ft_net','ide', 'idl']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = False

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        
        self.base.avgpool = nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.base.fc.in_features
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.base.fc = add_block

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            # print(name, module)
            if name == 'avgpool':
                break
            # print(x)
            x = module(x)

        pool5 = F.avg_pool2d(x, x.size()[2:])
        pool5 = pool5.view(pool5.size(0), -1)

        return pool5

class ft_net(nn.Module):

    def __init__(self, num_classes, num_features):
        super(ft_net,self).__init__()
        self.memorybank = torch.zeros(2, 1 * num_classes, 2048).cuda()
        self.memorybank = Parameter(self.memorybank)
        
        for p in self.parameters():
            p.requires_grad = False

        model_ft = torchvision.models.resnet50(pretrained=True)
        self.model = model_ft

        feature = []
        feature += [nn.Linear(2048, num_features)]
        feature += [nn.BatchNorm1d(num_features)]
        feature = nn.Sequential(*feature)
        feature = feature.apply(weights_init_kaiming)
        self.feature = feature

        classifier = []
        classifier += [nn.Linear(num_features, num_features // 2)]
        classifier += [nn.BatchNorm1d(num_features // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(num_features // 2, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        for name, module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        feature = F.max_pool2d(x, x.size()[2:])
        feature = feature.view(feature.size(0), -1)
        feature = self.feature(feature)

        x = self.classifier(feature)

        return x, feature

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, num_features, norm=False):
        super(Discriminator, self).__init__()
        self.norm = norm
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Linear(num_features, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        add_block += [nn.Linear(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

        classifier = []
        classifier += [nn.Linear(500, 2)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

def resnet18(**kwargs):
    Dis = Discriminator()
    return ResNet(18, **kwargs), Dis

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    # Dis = Discriminator()
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)

def ide(**kwargs):
    return ft_net(**kwargs), Discriminator(num_features=2048)

def idl(**kwargs):
    return ResNet(50, **kwargs)
