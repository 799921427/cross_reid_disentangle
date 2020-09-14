import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        #return self.op(inputs)
        return self.relu(self.bn(self.op(inputs)))

#class ProjectorFC(nn.Module):
#    def __init__(self, in_features, out_features):
#        super(ProjectorFC, self).__init__()
#        #self.op = nn.Linear(in_channels=in_features, out_channels=out_features, bias=False)
#        self.out_features = out_features
#        self.in_features = in_features
#        self.op = nn.Linear(self.in_features, self.out_features)
#    def forward(self, inputs):
#        N, C, W, H = inputs.shape
#        return self.op(inputs.view(N,-1)).view(N, self.out_features, W, H)

class ProjectorFC(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorFC, self).__init__()
        #self.op = nn.Linear(in_channels=in_features, out_channels=out_features, bias=False)
        self.out_features = out_features
        self.in_features = in_features
        #self.bn = nn.BatchNorm2d(self.in_features)
        #self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.5)
        self.op = nn.Linear(self.in_features, self.out_features)
    def forward(self, inputs):
        N, C, W, H = inputs.shape
        #inputs = self.dp(self.relu(self.bn(inputs)))
        inputs = self.dp(inputs)
#         pdb.set_trace()
        return self.op(inputs.view(N,-1)).view(N, self.out_features, W, H)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True, featuremap=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.featuremap = featuremap
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, l, g):
        N, C, W, H = l.size()
        # print(N,C,W,H)
        c = self.bn(self.op(l+g)) # batch_sizex1xWxH 
        # pdb.set_trace()
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)

        if self.featuremap == True:
            # print(g.size())
            return c.view(N,1,W, H), g
        else:
                
            if self.normalize_attn:
                # g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
                g = g.view(N,C,-1).sum(dim=2)/W/H # batch_sizexC
                # print(g.size())
                
            else:
                g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
            # print(g.size())
            return c.view(N,1,W,H), g


#class LinearAttentionBlock2(nn.Module):
#    def __init__(self, in_features, is_u_using = True):
#        super(LinearAttentionBlock2, self).__init__()
#        self.is_u_using = is_u_using
#        self.u = Variable(torch.empty(1,in_features,1,1), requires_grad=True).to(device)
#        self.bn = nn.BatchNorm2d(in_features)
#        nn.init.uniform_(self.u)
#        #nn.init.xavier_normal_(self.u)
#    def forward(self, l, g):
#        N, C, W, H = l.size()

#        if self.is_u_using == True:
#            c = l + g
#            d = self.u.expand_as(c) * c
#        else:
#            d = g.expand_as(l) * l
#        e = d.sum(dim=1)
#        s = F.softmax(e.view(N,1,-1), dim=2).view(N,1,W,H)
#        #g = (s.expand_as(l) * l).sum(dim=[-1,-2])
#        g = (self.bn(s.expand_as(l) * l)).sum(dim=[-1,-2])/W/H
#        return e.view(N,1,W,H), g


class LinearAttentionBlock2(nn.Module):
    def __init__(self, in_features, is_u_using = True,  featuremap=True):
        super(LinearAttentionBlock2, self).__init__()
        self.is_u_using = is_u_using
        self.featuremap = featuremap
        self.u = Variable(torch.empty(1,in_features,1,1), requires_grad=True).to(device)
        self.bn = nn.BatchNorm2d(in_features)
        self.bn_for_l = nn.BatchNorm2d(in_features)
        self.relu_for_l = nn.ReLU(inplace=True)
        nn.init.uniform_(self.u)
        #nn.init.xavier_normal_(self.u)
        self.conv = nn.Conv2d(in_features, 1, kernel_size=1, stride=1,bias=False)

        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l = self.relu_for_l(self.bn_for_l(l)) # bn before relu?
        if self.is_u_using == True:
            c = l + g
            #d = self.u.expand_as(c) * c
            d = self.conv(c)
            e = d
        else:
            d = g.expand_as(l) * l
            #e = d.sum(dim=1)
            d = self.conv(d)
            e = d
        e = self.relu2(self.bn2(e))
        s = F.softmax(e.view(N,1,-1), dim=2).view(N,1,W,H)
        g_t = s.expand_as(l) * l
        # print(g_t.size())
        g = (s.expand_as(l) * l).sum(-1).sum(-1)
        # print(g.size())
        #g = (self.bn(s.expand_as(l) * l)).sum(dim=[-1,-2])/W/H
        if self.featuremap:
            out = g_t
        else:
            out = g
        return e.view(N,1,W,H), g_t, g


'''
Grid attention block

Reference papers
Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114

Reference code
https://github.com/ozan-oktay/Attention-Gated-Networks
'''
class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output
