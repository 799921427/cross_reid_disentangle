from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['AdaINGen']

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, fp16):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        n_downsample = 2
        # decoder res number
        n_res = 4
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        mlp_norm = params['mlp_norm']
        id_dim = params['id_dim']
        which_dec = params['dec']
        dropout = params['dropout']
        tanh = params['tanh']
        non_local = params['non_local']
        #
        # # content encoder
        # self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type,
        #                                   dropout=dropout, tanh=tanh, res_type='basic')

        self.output_dim = self.enc_content.output_dim
        if which_dec == 'basic':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ,
                               pad_type=pad_type, res_type='basic', non_local=non_local, fp16=fp16)
        elif which_dec == 'slim':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ,
                               pad_type=pad_type, res_type='slim', non_local=non_local, fp16=fp16)
        elif which_dec == 'series':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ,
                               pad_type=pad_type, res_type='series', non_local=non_local, fp16=fp16)
        elif which_dec == 'parallel':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ,
                               pad_type=pad_type, res_type='parallel', non_local=non_local, fp16=fp16)
        else:
            ('unkonw decoder type')

        # MLP to generate AdaIN parameters
        self.mlp_w1 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w2 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w3 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w4 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)

        self.mlp_b1 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b2 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b3 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b4 = MLP(id_dim, 2 * self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)

        self.apply(weights_init(params['init']))

    # def encode(self, images):
    #     # encode an image to its content and style codes
    #     content = self.enc_content(images)
    #     return content

    def decode(self, content, ID):
        # decode style codes to an image
        ID1 = ID[:, :2048]
        ID2 = ID[:, 2048:4096]
        ID3 = ID[:, 4096:6144]
        ID4 = ID[:, 6144:]
        adain_params_w = torch.cat((self.mlp_w1(ID1), self.mlp_w2(ID2), self.mlp_w3(ID3), self.mlp_w4(ID4)), 1)
        adain_params_b = torch.cat((self.mlp_b1(ID1), self.mlp_b2(ID2), self.mlp_b3(ID3), self.mlp_b4(ID4)), 1)
        self.assign_adain_params(adain_params_w, adain_params_b, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:, :dim].contiguous()
                std = adain_params_w[:, :dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1) > dim:  # Pop the parameters
                    adain_params_b = adain_params_b[:, dim:]
                    adain_params_w = adain_params_w[:, dim:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += m.num_features
        return num_adain_params



class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='adain', activ='relu', pad_type='zero',
                 res_type='basic', non_local=False, fp16=False):
        super(Decoder, self).__init__()
        self.input_dim = dim
        self.model = []
        self.model += [nn.Dropout(p=dropout)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, res_type=res_type)]
        # non-local
        if non_local > 0:
            self.model += [NonlocalBlock(dim)]
            print('use non-local!')
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type,
                                       fp16=fp16)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if res_type == 'basic' or res_type == 'nonlocal':
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'slim':
            dim_half = dim // 2
            model += [Conv2dBlock(dim, dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        elif res_type == 'series':
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'parallel':
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type == 'nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out



def AdaINGen(**kwargs):
    return AdaINGen(**kwargs)


def Discriminator(**kwargs):
    return discriminator(**kwargs)


def Classifier(**kwargs):
    return classifier(**kwargs)