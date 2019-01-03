import logging
import torch.nn as nn


from ..utils import conv1x1_group, conv3x3_group, \
    norm_layer, kaiming_init, constant_init, load_checkpoint


class Conv_dw_pw(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 widthmultiplier=1):
        super(Conv_dw_pw, self).__init__()
        self.inplanes = round(widthmultiplier * inplanes)
        self.planes = round(widthmultiplier * planes)
        self.dw_conv = conv3x3_group(self.inplanes, self.inplanes,
                                     stride=stride, dilation=dilation,
                                     groups=self.inplanes)
        self.pw_conv = conv1x1_group(self.inplanes, self.planes)

        # we want to load pre-trained models
        # for keep the layer name the same as pre-trained models
        norm_layers = [norm_layer(self.inplanes, use_gn),
                       norm_layer(self.planes, use_gn)]
        self.norm_names = ['bn1', 'bn2'] if not use_gn else ['gn1', 'gn2']
        for (name, layer) in zip(self.norm_names, norm_layers):
            self.add_module(name, layer)

    def forward(self, x):
        # dw
        x = self.dw_conv(x)
        norm = getattr(self, self.norm_names[0])
        x = norm(x)
        x = nn.ReLU(x)

        # pw
        x = self.pw_conv(x)
        norm = getattr(self, self.norm_names[0])
        x = norm(x)
        x = nn.ReLU(x)
        return x


def _make_layers(block,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 widthmultiplier=1):
    layers = [
        block(
            inplanes,
            planes,
            stride,
            dilation,
            use_gn)
    ]
    return nn.Sequential(*layers)


class MobileNets(nn.Module):
    r"""PyTorch implementation of the MobileNets architecture
    <https://arxiv.org/abs/1704.04861>`_.
    Model has been designed to work on either ImageNet or CIFAR-10
    Args:
        num_classes (int): 1000 for ImageNet, 10 for CIFAR-10
        large_img (bool): True for ImageNet, False for CIFAR-10
    """

    # (inplanes, planes, stride, dilation)
    arch_settings = [[(32, 64, 1, 1)],
                     [(64, 128, 2, 1), (128, 128, 1, 1)],
                     [(128, 256, 2, 1), (256, 256, 1, 1)],
                     [(256, 512, 2, 1), (512, 512, 1, 1),
                      (512, 512, 1, 1), (512, 512, 1, 1),
                      (512, 512, 1, 1), (512, 512, 1, 1)],
                     [(512, 1024, 2, 1), (1024, 1024, 1, 1)]]

    def __init__(self,
                 num_stages=13,
                 out_indices=(0, 1, 2, 3, 4),
                 forzen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_forzen=False,
                 widthmultiplier=1):
        super(MobileNets, self).__init__()

        assert 1 <= num_stages <= 13

        self.conv1 = conv3x3_group(in_planes=3,
                                   out_planes=round(32 * widthmultiplier),
                                   stride=2)
        self.norm1_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm1_name, norm_layer(32, use_gn))

        stage_blocks = self.arch_settings[:num_stages]
        self.mobilev1_layers = []
        for i, stage_arch in enumerate(stage_blocks):
            for j, (inplanes, planes, stride, dilation) in enumerate(
                    stage_arch):
                mobilev1_layer = _make_layers(
                    Conv_dw_pw,
                    inplanes,
                    planes,
                    stride,
                    dilation,
                    use_gn,
                    widthmultiplier)
                layer_name = 'layer{}'.format(i + 1)
                self.add_module(layer_name, mobilev1_layer)
                self.mobilev1_layers.append(layer_name)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        norm1 = getattr(self, self.norm1_name)
        x = norm1(x)
        x = nn.ReLU(x)

        outs = []
        for i, layer_name in enumerate(self.mobilev1_layers):
            mobilev1_layer = getattr(self, layer_name)
            x = mobilev1_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(MobileNets, self).train(mode)
        if not self.use_gn:
            if self.bn_eval:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        if self.bn_frozen:
                            for params in m.parameters():
                                params.requires_grad = False

        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            norm1 = getattr(self, self.norm_name)
            for param in norm1.parameters():
                param.requires_grad = False
            norm1.eval()
            norm1.weights.requires_grad = False
            norm1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
