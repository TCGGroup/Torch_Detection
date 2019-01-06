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
                 use_gn=False):
        super(Conv_dw_pw, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
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

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.use_gn = use_gn

    def forward(self, x):
        # dw
        x = self.dw_conv(x)
        norm1 = getattr(self, self.norm_names[0])
        x = norm1(x)
        x = self.relu(x)

        # pw
        x = self.pw_conv(x)
        norm2 = getattr(self, self.norm_names[0])
        x = norm2(x)
        x = self.relu(x)
        return x


def _make_layers(block,
                 inplanes,
                 planes,
                 blocks,
                 stride=1,
                 dilation=1,
                 use_gn=False):
    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            use_gn=use_gn)
    )
    inplanes = planes
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                stride=1,
                dilation=dilation,
                use_gn=use_gn)
        )
    return nn.Sequential(*layers)


class MobileNet(nn.Module):
    """
    MobileNet backbone.
    paper: https://arxiv.org/abs/1704.04861

    Args:
        width_multi (float): width_multi of mobilenet, from {0.25, 0.5, 0.75,
            1.0}.
        num_stages (int): mobilenet stages, normally 5.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        use_gn (bool): Whether to use GN for normalization layers
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var), only used when using BN.
        bn_frozen (bool): Whether to freeze weight and bias of BN layers, only
            used when using BN.
    """

    arch_settings = {
        0.25: ((16, 32, 64, 128, 256), (1, 2, 2, 6, 2)),
        0.5: ((32, 64, 128, 256, 512), (1, 2, 2, 6, 2)),
        0.75: ((48, 96, 172, 384, 768), (1, 2, 2, 6, 2)),
        1.0: ((64, 128, 256, 512, 1024), (1, 2, 2, 6, 2))
    }

    def __init__(self,
                 width_multi=1.0,
                 num_stages=5,
                 strides=(1, 2, 2, 2, 2),
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(MobileNet, self).__init__()
        if width_multi not in self.arch_settings:
            raise KeyError('invalid depth {} for mobilenet'.format(
                width_multi))
        assert 1 <= num_stages <= 5
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        if not use_gn:
            self.bn_eval = bn_eval
            self.bn_frozen = bn_frozen
        self.use_gn = use_gn

        self.inplanes = round(32 * width_multi)
        self.conv1 = conv3x3_group(3, round(32 * width_multi), stride=2)
        self.norm1_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm1_name,
                        norm_layer(round(32 * width_multi), use_gn))
        self.relu = nn.ReLU(inplace=True)

        stage_outplanes, stage_blocks = self.arch_settings[width_multi]
        stage_outplanes = stage_outplanes[:num_stages]
        stage_blocks = stage_blocks[:num_stages]

        self.mobilev1_layers = []
        for i, num_block in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = stage_outplanes[i]

            mobilev1_layer = _make_layers(
                Conv_dw_pw,
                self.inplanes,
                planes,
                num_block,
                stride=stride,
                dilation=dilation,
                use_gn=use_gn)
            self.inplanes = planes
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
        x = self.relu(x)

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
        super(MobileNet, self).train(mode)
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
