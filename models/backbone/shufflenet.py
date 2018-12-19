import logging

import torch
import torch.nn as nn

from ..utils import conv1x1_group, conv3x3_group, norm_layer, \
    kaiming_init, constant_init, load_checkpoint


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        """
        Channel shuffle: [N, C, H, W] -> [N, g, C/g, H, W] ->
                         [N, C/g, g, H, W] -> [N, C, H, W]
        """
        N, C, H, W = x.size()

        g = self.groups
        return x.view(N, g, C / g, H, W).permute(
            0, 2, 1, 3, 4).reshape(x.size())


class ShuffleNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 outplanes,
                 groups,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 downsample=None):
        super(ShuffleNetBottleneck, self).__init__()

        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        g = 1 if inplanes == 24 else groups
        # NOTE: ensure output of concat matches the output channels
        planes = outplanes // self.expansion
        outplanes = outplanes - inplanes if stride == 2 else outplanes
        self.conv1 = conv1x1_group(inplanes, planes, groups=g)
        self.shuffle1 = ShuffleLayer(groups=g)
        self.conv2 = conv3x3_group(
            planes, planes, stride=stride, dilation=dilation, groups=planes)
        self.conv3 = conv1x1_group(planes, outplanes, groups=groups)

        # we want to load pre-trained models
        # for keep the layer name the same as pre-trained models
        norm_layers = []
        norm_layers.append(norm_layer(planes, use_gn))
        norm_layers.append(norm_layer(planes, use_gn))
        norm_layers.append(norm_layer(outplanes, use_gn))
        self.norm_names = ['bn1', 'bn2', 'bn3'] \
            if not use_gn else ['gn1', 'gn2', 'gn3']
        for name, layer in zip(self.norm_names, norm_layers):
            self.add_module(name, layer)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.use_gn = use_gn

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        norm1 = getattr(self, self.norm_names[0])
        out = norm1(out)
        out = self.relu(out)
        out = self.shuffle1(out)

        out = self.conv2(out)
        norm2 = getattr(self, self.norm_names[1])
        out = norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        norm3 = getattr(self, self.norm_names[2])
        out = norm3(out)

        if self.stride == 2 and self.downsample is not None:
            residual = self.downsample(x)
            out = torch.cat((out, residual), 1)
        else:
            out += residual

        out = self.relu(out)
        return out


def _make_shuffle_stage(block,
                        inplanes,
                        outplanes,
                        blocks,
                        groups,
                        stride=1,
                        dilation=1,
                        use_gn=False):
    downsample = None
    if stride != 1:
        downsample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    layers = []
    layers.append(
        block(inplanes,
              outplanes,
              groups,
              stride=stride,
              dilation=dilation,
              use_gn=use_gn,
              downsample=downsample))
    inplanes = outplanes
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  outplanes,
                  groups,
                  stride=1,
                  dilation=dilation,
                  use_gn=use_gn))
    return nn.Sequential(*layers)


class ShuffleNet(nn.Module):
    """
    ShuffleNet backbone.
    paper: https://arxiv.org/abs/1707.01083

    The main difference between ResNeXt and shuffleNet is using group in the
    first conv1x1 of a bottleneck, adding shuffle layer after the grouped 1x1
    conv, changing the group 3x3 conv into depthwise convolution.

    Args:
        groups (int): Groups used in 1x1 conv, from {1, 2, 3, 4, 8}.
        num_stages (int): ShuffleNet stages, normally 3.
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
    # the settings for different groups, the format is:
    # groups: (stage_out_planes, stage_num_blocks)
    arch_settings = {
        1: ((144, 288, 576), (4, 8, 4)),
        2: ((200, 400, 800), (4, 8, 4)),
        3: ((240, 480, 960), (4, 8, 4)),
        4: ((272, 544, 1088), (4, 8, 4)),
        8: ((384, 768, 1536), (4, 8, 4))
    }

    def __init__(self,
                 groups,
                 num_stages=3,
                 strides=(2, 2, 2),
                 dilations=(1, 1, 1),
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(ShuffleNet, self).__init__()
        if groups not in self.arch_settings:
            raise KeyError('invalid groups number {} '
                           'for shuffleNet'.format(groups))

        assert 1 <= num_stages <= 3
        stage_outplanes, stage_blocks = self.arch_settings[groups]
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        if not use_gn:
            self.bn_eval = bn_eval
            self.bn_frozen = bn_frozen
        self.use_gn = use_gn

        self.inplanes = 24
        # stage1
        self.conv1 = conv3x3_group(3, 24, stride=2)
        self.norm_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm_name, norm_layer(64, use_gn))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.shuffle_stages = []
        for i, num_blocks in enumerate(stage_blocks):
            outplanes = stage_outplanes[i]
            stride = strides[i]
            dilation = dilations[i]
            shuffle_stage = _make_shuffle_stage(
                ShuffleNetBottleneck,
                inplanes=self.inplanes,
                outplanes=outplanes,
                blocks=num_blocks,
                groups=groups,
                stride=stride,
                dilation=dilation,
                use_gn=use_gn)
            self.inplanes = outplanes
            stage_name = 'stage{}'.format(i + 2)
            self.add_module(stage_name, shuffle_stage)
            self.shuffle_stages.append(stage_name)

        self.feat_dim = stage_outplanes[-1]

    def forward(self, x):
        x = self.conv1(x)
        norm1 = getattr(self, self.norm_name)
        x = norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, stage_name in enumerate(self.shuffle_stages):
            shuffle_stage = getattr(self, stage_name)
            x = shuffle_stage(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

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

    def train(self, mode=True):
        super(ShuffleNet, self).train(mode)
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
            norm1.weight.requires_grad = False
            norm1.bias.requires_grad = False
            for i in range(2, self.frozen_stages + 1):
                mod = getattr(self, 'stage{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False