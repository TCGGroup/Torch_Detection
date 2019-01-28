import logging
import torch.nn as nn

from ..utils import ConvModule, kaiming_init, constant_init, load_checkpoint
from ..registry import BACKBONES


def _make_vgg_layer(block,
                    inplanes,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    normalize=None,
                    use_gn=False,
                    activation='relu',
                    maxpool=True):
    layers = []
    with_bias = normalize is not None
    layers.append(
        block(inplanes,
              planes,
              kernel_size=3,
              stride=stride,
              padding=dilation,
              dilation=dilation,
              bias=with_bias,
              normalize=normalize,
              use_gn=use_gn,
              activation=activation)
    )
    inplanes = planes
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  planes,
                  kernel_size=3,
                  stride=1,
                  padding=dilation,
                  dilation=dilation,
                  bias=with_bias,
                  normalize=normalize,
                  use_gn=use_gn,
                  activation=activation))
    if maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


@BACKBONES.register_module
class VGG(nn.Module):

    """
    vgg backbone.
    paper: https://arxiv.org/pdf/1409.1556.pdf

    Args:
        depth (int):vgg depth, range from {11, 13, 16, 19}.
        num_stages (int): vgg stages, normally 5.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (bool, optional): whether to use normalization layer.
        use_gn (bool): Whether to use GN for normalization layers
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var), only used when using BN.
        bn_frozen (bool): Whether to freeze weight and bias of BN layers, only
            used when using BN.
    """

    arch_settings = {
        11: (ConvModule, (1, 1, 2, 2, 2)),
        13: (ConvModule, (2, 2, 2, 2, 2)),
        16: (ConvModule, (2, 2, 3, 3, 3)),
        19: (ConvModule, (2, 2, 4, 4, 4)),
    }

    def __init__(self,
                 depth,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 normalize=None,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for vgg'.format(depth))

        assert 1 <= num_stages <= 5
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.activation = 'relu' if self.normalize is not None else None
        if not use_gn:
            self.bn_eval = bn_eval
            self.bn_frozen = bn_frozen
        self.use_gn = use_gn

        self.inplanes = 3

        self.vgg_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = max(64 * 2 ** i, 512)
            vgg_layer = _make_vgg_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                normalize=self.normalize,
                use_gn=self.use_gn,
                activation=self.activation,
                maxpool=True)
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, vgg_layer)
            self.vgg_layers.append(layer_name)

        self.feat_dim = 512

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
        outs = []
        for i, layer_name in enumerate(self.vgg_layers):
            vgg_layer = getattr(self, layer_name)
            x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if not self.use_gn:
            if self.bn_eval:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        if self.bn_frozen:
                            for params in m.parameters():
                                params.requires_grad = False

        if mode and self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
