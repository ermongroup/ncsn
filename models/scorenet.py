import torch.nn as nn
import functools
import torch
from torchvision.models import ResNet
import torch.nn.functional as F
from .pix2pix import init_net, UnetSkipConnectionBlock, get_norm_layer, init_weights, ResnetBlock, \
    UnetSkipConnectionBlockWithResNet


class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)

        if not resize:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)


class DeconvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, True)

        if not resize:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)


class ResScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'elu'
        self.convs = nn.Sequential(
            nn.Conv2d(3, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, resize=True, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act),
        )

        self.deconvs = nn.Sequential(
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(4 * self.ndf, 4 * self.ndf, act=act),
            DeconvResBlock(4 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            DeconvResBlock(2 * self.ndf, self.ndf, resize=True, act=act),
            DeconvResBlock(self.ndf, self.ndf, act=act),
            nn.Conv2d(self.ndf, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = 2 * x - 1.
        res = self.deconvs(self.convs(x))
        return res


class ResNetScore(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, config):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()

        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf * 2
        n_blocks = 6
        norm_layer = get_norm_layer('instance')
        use_dropout = False
        padding_type = 'reflect'
        assert (n_blocks >= 0)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ELU()]

        n_downsampling = 1
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ELU()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ELU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        input = 2 * input - 1.
        return self.model(input)


class UNetResScore(nn.Module):
    def __init__(self, config):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf
        self.config = config
        norm_layer = get_norm_layer('instance')
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
        #                                      innermost=True)  # add the innermost layer

        # for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
        #                                          norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlockWithResNet(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

        # init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, input):
        """Standard forward"""
        if not self.config.data.logit_transform:
            input = 2 * input - 1.
        return self.model(input)


class UNetScore(nn.Module):
    def __init__(self, config):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf
        self.config = config
        norm_layer = get_norm_layer('instance')
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
        #                                      innermost=True)  # add the innermost layer

        # for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
        #                                          norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer)
        if config.data.image_size == 32:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
        elif config.data.image_size == 16:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                 norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

        # init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, input):
        """Standard forward"""
        if not self.config.data.logit_transform:
            input = 2 * input - 1.
        return self.model(input)


class ResEnergy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'softplus'
        self.convs = nn.Sequential(
            nn.Conv2d(1, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act)
        )

    def forward(self, x):
        x = 2 * x - 1.
        res = self.convs(x)
        res = res.view(res.shape[0], -1).mean(dim=-1)
        return res


class MLPScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main = nn.Sequential(
            nn.Linear(10 * 10, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 100),
            nn.LayerNorm(100)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if x.is_cuda and self.config.training.ngpu > 1:
            score = nn.parallel.data_parallel(
                self.main, x, list(range(self.config.training.ngpu)))
        else:
            score = self.main(x)

        return score.view(x.shape[0], 1, 10, 10)


class LargeScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.data.channels, nef, 16, stride=2, padding=2),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 28 x 28
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 28 * 28, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * 28 * 28)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 28, 28)
        return score


class Score(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.data.channels, nef, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 28 x 28
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 28 * 28, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * 28 * 28)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 28, 28)
        return score


class SmallScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef * 4
        self.u_net = nn.Sequential(
            # input is (nc) x 10 x 10
            nn.Conv2d(config.data.channels, nef, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 6 x 6
            nn.Conv2d(nef, nef * 2, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 6 x 6
            nn.ConvTranspose2d(nef * 2, nef, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 6 x 6
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 10 ** 2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, config.data.channels * 10 ** 2)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 10, 10)
        return score
