import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .blocks import ProjectorBlock, ProjectorFC, LinearAttentionBlock, LinearAttentionBlock2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def linear_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_Classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class AttentionModule(nn.Module):
    def __init__(self, num_features, num_classes, attention_mode):
        super(AttentionModule, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.attention_mode = attention_mode
        if attention_mode == 1:
            self.projector1 = ProjectorBlock(256, num_features)
            self.projector2 = ProjectorBlock(512, num_features)
            self.projector3 = ProjectorBlock(1024, num_features)
            self.projector4 = ProjectorBlock(2048, num_features)
            self.att1 = LinearAttentionBlock(in_features=num_features, normalize_attn=True, featuremap=False)
            self.att2 = LinearAttentionBlock(in_features=num_features, normalize_attn=True, featuremap=False)
            self.att3 = LinearAttentionBlock(in_features=num_features, normalize_attn=True, featuremap=False)
            self.att4 = LinearAttentionBlock(in_features=num_features, normalize_attn=True, featuremap=False)
            # self.att_classifier1 = nn.Linear(in_features=256, out_features=num_classes, bias=True)
            # self.att_classifier2 = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            # self.att_classifier3 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            self.att_classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        if attention_mode == 2 or attention_mode == 3:
            if attention_mode == 2: is_u_using = True
            if attention_mode == 3: is_u_using = False
            self.projector1 = ProjectorFC(num_features, 256)
            self.projector2 = ProjectorFC(num_features, 512)
            self.projector3 = ProjectorFC(num_features, 1024)
            self.projector4 = ProjectorFC(num_features, 2048)
            self.att1 = LinearAttentionBlock2(in_features=256, is_u_using=is_u_using)
            self.att2 = LinearAttentionBlock2(in_features=512, is_u_using=is_u_using)
            self.att3 = LinearAttentionBlock2(in_features=1024, is_u_using=is_u_using)
            self.att4 = LinearAttentionBlock2(in_features=2048, is_u_using=is_u_using)

            self.att_classifier = nn.Linear(in_features=256+512+1024+2048, out_features=num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        # self.att_classifier.apply(weights_init_classifier)
        self.att_classifier.apply(weights_init_classifier)
        self.projector1.apply(weights_init_kaiming)
        self.projector2.apply(weights_init_kaiming)
        self.projector3.apply(weights_init_kaiming)
        self.projector4.apply(weights_init_kaiming)
        self.att1.apply(linear_init_kaiming)
        self.att2.apply(linear_init_kaiming)
        self.att3.apply(linear_init_kaiming)
        self.att4.apply(linear_init_kaiming)

    def forward(self, feature_map, g):
        # print(feature_map[0].size())
        if self.attention_mode == 1:
            fea1, g1 = self.att1(self.projector1(feature_map[0]), g)
            fea2, g2 = self.att2(self.projector2(feature_map[1]), g)
            fea3, g3 = self.att3(self.projector3(feature_map[2]), g)
            fea4, g4 = self.att4(self.projector4(feature_map[3]), g)
            g = torch.cat((g1, g2, g3, g4), dim=1)
            # print(g1.size())
            # print(g2.size())
            # print(g3.size())
            # print(g4.size())
            cls_att=self.att_classifier(g1)+self.att_classifier(g2)+self.att_classifier(g3)+self.att_classifier(g4)
            cls_att = cls_att / 4
            # cls_att1 = self.att_classifier(g1)
            # cls_att2 = self.att_classifier(g2)
            # cls_att3 = self.att_classifier(g3)
            # cls_att4 = self.att_classifier(g4)
            # cls_att = (cls_att1 + cls_att2 + cls_att3 + cls_att4) / 4
            # print(fea1.size())
            # print(fea2.size())
            # print(fea3.size())
            # print(fea4.size())
            # att_feats = [g1, g2, g3, g4]
            att_feats = [fea1, fea2, fea3, fea4]
        elif self.attention_mode == 2 or self.attention_mode == 3:
            fea1, g1, g11 = self.att1(feature_map[0],self.projector1(g))
            fea2, g2, g22 = self.att2(feature_map[1], self.projector2(g))
            fea3, g3, g33 = self.att3(feature_map[2], self.projector3(g))
            fea4, g4, g44 = self.att4(feature_map[3], self.projector4(g))
            # print(fea1.size())
            # [b, 1, 64, 32]
            # [b, 1, 32, 16]
            # [b, 1, 16, 8]
            # [b, 1, 16, 8]
            # print(g1.size())
            # [b, 256, 64, 32]
            # print(g11.size())
            # [b, 256]
            # g = g44
            g = torch.cat((g11,g22,g33,g44), dim = 1)
            # print(g.size())
            #[b, 3840]
            # att_feats = [g1,g2,g3,g4]
            att_feats = [fea1, fea2, fea3, fea4]
            # print(att_feats.shape)
            cls_att = self.att_classifier(g)
            # print(cls_att.size)

        return g, cls_att, att_feats