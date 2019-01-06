import logging
import torch.nn as nn

from ..utils import conv1x1_group, conv3x3_group, \
    norm_layer, kaiming_init, constant_init, load_checkpoint


class LinearBottleNeck(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 expansion=6,
                 stride=1,
                 dilation=1,
                 use_gn=False):
        super(LinearBottleNeck, self).__init__()

        self.planes = inplanes * expansion

        self.conv1 = conv1x1_group(inplanes, self.planes)
        self.conv2 = conv3x3_group(self.planes, self.planes, stride=stride,
                                   dilation=dilation, groups=self.planes)
        self.conv3 = conv1x1_group(self.planes, outplanes)

        norm_layers = [norm_layer(self.planes, use_gn),
                       norm_layer(self.planes, use_gn),
                       norm_layer(outplanes, use_gn)]
        self.norm_names = ['bn1', 'bn2', 'bn3'] \
            if not use_gn else ['gn1', 'gn2', 'gn3']
        for name, layer in zip(self.norm_names, norm_layers):
            self.add_module(name, layer)

        self.stride = stride
        self.dialtion = dilation
        self.use_gn = use_gn
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        norm1 = getattr(self, self.norm_names[0])
        out = norm1(out)
        out = self.relu6(out)

        out = self.conv2(out)
        norm2 = getattr(self, self.norm_names[1])
        out = norm2(out)
        out = self.relu6(out)

        out = self.conv3(out)
        norm3 = getattr(self, self.norm_names[2])
        out = norm3(out)

        if self.stride == 1 & self.inplanes == self.planes:
            out += x
        return out


def _make_layers(block,
                 inplanes,
                 outplanes,
                 blocks,
                 expansion=6,
                 stride=1,
                 dilation=1,
                 use_gn=False):
    layers = [
        block(inplanes,
              outplanes,
              expansion=expansion,
              stride=stride,
              dilation=dilation,
              use_gn=use_gn)
    ]

    inplanes = outplanes
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  outplanes,
                  expansion=expansion,
                  stride=1,
                  dilation=dilation,
                  use_gn=use_gn)
        )
    return nn.Sequential(*layers)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone.
    paper: https://arxiv.org/pdf/1801.04381.pdf

    Args:
        num_stages (int): Resnet stages, normally 4.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        use_gn (bool): Whether to use GN for normalization layers
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var), only used when using BN.
        bn_frozen (bool): Whether to freeze weight and bias of BN layers, only
            used when using BN.
    """

    # (expansion, out_planes, num_blocks, stride, dilation)
    arch_settings = [(1, 16, 1, 1, 1),
                     (6, 24, 2, 2, 1),  # NOTE: change stride 2->1 for CIFAR10
                     (6, 32, 3, 2, 1),
                     (6, 64, 4, 2, 1),
                     (6, 96, 3, 1, 1),
                     (6, 160, 3, 2, 1),
                     (6, 320, 1, 1, 1)]

    def __init__(self,
                 num_stages=7,
                 out_indices=(0, 1, 2, 3, 4, 5, 6),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(MobileNetV2, self).__init__()

        assert 1 <= num_stages <= 7
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.relu6 = nn.ReLU6(x)

        if not use_gn:
            self.bn_eval = bn_eval
            self.bn_frozen = bn_frozen
        self.use_gn = use_gn
        self.conv1 = conv3x3_group(
            in_planes=3, out_planes=32, stride=2, dilation=1)

        self.norm1_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm1_name, norm_layer(32, use_gn))

        stage_blocks = self.arch_settings[:num_stages]
        self.inplanes = 32
        self.mobilev2_layers = []
        for i, (expansion, outplanes, num_blocks, stride, dilation)\
                in enumerate(stage_blocks):
            mobilev2_layer = _make_layers(
                LinearBottleNeck,
                self.inplanes,
                outplanes,
                num_blocks,
                expansion,
                stride,
                dilation,
                use_gn=use_gn)
            self.inplanes = outplanes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, mobilev2_layer)
            self.mobilev2_layers.append(layer_name)

        self.conv2 = conv1x1_group(in_planes=320, out_planes=1280)
        self.norm2_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm2_name, norm_layer(1280, use_gn))

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
        x = self.relu6(x)

        outs = []
        for i, layer_name in enumerate(self.mobilev2_layers):
            mobilev2_layer = getattr(self, layer_name)
            x = mobilev2_layer(x)
            if i in self.out_indices:
                if i < 7:
                    outs.append(x)
                else:
                    x = self.conv2(x)
                    norm2 = getattr(self, self.norm2_name)
                    x = norm2(x)
                    x = self.relu6(x)
                    outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
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
                if i < 7:
                    mod = getattr(self, 'layer{}'.format(i))
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False
                else:
                    for param in self.conv2.parameters():
                        param.requires_grad = False
                    norm2 = getattr(self, self.norm2_name)
                    for param in norm2.parameters():
                        param.requires_grad = False
                    norm2.eval()
                    norm2.weight.require_grad = False
                    norm2.bias.requires_grad = False
