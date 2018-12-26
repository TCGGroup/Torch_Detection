import logging

import torch
import torch.nn as nn

from ..utils import conv1x1_group, conv3x3_group, norm_layer, \
    ShuffleLayer, ChannelSplit, kaiming_init, constant_init, \
    load_checkpoint


def InvertedLayer(inplanes,
                  outplanes,
                  stride=1,
                  dilation=1,
                  use_gn=False,
                  type='right'):
    """
    This layer can be used in ShuffleNetv2.

    The main difference between this layer with InvertedResidual in
    MobileNetv2 is the use of ReLU layer, in this layer, ReLu layer for
    depthwise conv is removed.

    There are two types, `left` and `right`, for type `left`,  there are
    two conv layers: 3x3 dwconv + 1x1 conv, for type `right`,  there are
    three conv layers: 1x1 conv + 3x3 dwconv + 1x1 conv.
    """
    assert type in ['left', 'right']
    if type == 'left':
        return nn.Sequential(
            # dw
            conv3x3_group(inplanes,
                          inplanes,
                          stride=stride,
                          dilation=dilation,
                          groups=inplanes),
            norm_layer(inplanes, use_gn=use_gn),
            # pw-linear
            conv1x1_group(inplanes, outplanes),
            norm_layer(outplanes, use_gn=use_gn),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            # pw
            conv1x1_group(inplanes, outplanes),
            norm_layer(outplanes, use_gn=use_gn),
            nn.ReLU(inplace=True),
            # dw
            conv3x3_group(outplanes,
                          outplanes,
                          stride=stride,
                          dilation=dilation,
                          groups=outplanes),
            norm_layer(outplanes, use_gn=use_gn),
            # pw-linear
            conv1x1_group(outplanes, outplanes),
            norm_layer(outplanes, use_gn=use_gn),
            nn.ReLU(inplace=True)
        )


class ShuffleNetv2Bottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 downsample=None):
        super(ShuffleNetv2Bottleneck, self).__init__()
        assert self.expansion == 2
        assert stride in [1, 2]
        # NOTE: ensure output of concat matches the output channels
        planes = outplanes // self.expansion

        self.branch = InvertedLayer(inplanes,
                                    planes,
                                    stride=stride,
                                    dilation=dilation,
                                    use_gn=use_gn,
                                    type='right')
        self.channel_split = ChannelSplit()
        self.shuffle = ShuffleLayer(groups=self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.use_gn = use_gn

    def forward(self, x):
        left, right = self.channel_split(x)

        right = self.branch(right)
        if self.stride == 2 and self.downsample is not None:
            left = self.downsample(left)
        out = torch.cat((left, right), 1)
        out = self.shuffle(out)
        return out


def _make_shufflev2_stage(block,
                          inplanes,
                          outplanes,
                          blocks,
                          stride=1,
                          dilation=1,
                          use_gn=False):
    downsample = None
    if stride != 1:
        planes = outplanes // block.expansion
        downsample = InvertedLayer(inplanes,
                                   planes,
                                   stride=stride,
                                   use_gn=use_gn,
                                   type='left')

    layers = []
    layers.append(
        block(inplanes,
              outplanes,
              stride=stride,
              dilation=dilation,
              use_gn=use_gn,
              downsample=downsample))
    inplanes = outplanes
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  outplanes,
                  stride=1,
                  dilation=dilation,
                  use_gn=use_gn))
    return nn.Sequential(*layers)


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 backbone.
    paper: https://arxiv.org/pdf/1807.11164.pdf

    There are four guidelines in the paper for design efficient and effective
    model. Referring the paper for details.
    Use ShuffleNetV2 as a backbone in object detection, for the last stage,
    we use the last 1x1 conv's output feature maps.

    Args:
        width_mult (float): Width multiple times for ShuffleNetv2,
            from {0.5, 1.0, 1.5, 2.0}.
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
        0.5: ((48, 96, 192, 1024), (4, 8, 4)),
        1.0: ((116, 232, 464, 1024), (4, 8, 4)),
        1.5: ((176, 352, 704, 1024), (4, 8, 4)),
        2.0: ((244, 488, 976, 2048), (4, 8, 4)),
    }

    def __init__(self,
                 width_mult,
                 num_stages=3,
                 strides=(2, 2, 2),
                 dilations=(1, 1, 1),
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(ShuffleNetV2, self).__init__()
        if width_mult not in self.arch_settings:
            raise KeyError('invalid width multiple times {} '
                           'for shuffleNetV2'.format(width_mult))

        assert 1 <= num_stages <= 3
        stage_outplanes, stage_blocks = self.arch_settings[width_mult]
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
        self.norm_name1 = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm_name1, norm_layer(24, use_gn))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2-4
        self.shuffle_stages = []
        for i, num_blocks in enumerate(stage_blocks):
            outplanes = stage_outplanes[i]
            stride = strides[i]
            dilation = dilations[i]
            shuffle_stage = _make_shufflev2_stage(
                ShuffleNetv2Bottleneck,
                inplanes=self.inplanes,
                outplanes=outplanes,
                blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                use_gn=use_gn)
            self.inplanes = outplanes
            stage_name = 'stage{}'.format(i + 2)
            self.add_module(stage_name, shuffle_stage)
            self.shuffle_stages.append(stage_name)

        # stage5
        self.conv5 = conv1x1_group(self.inplanes, stage_outplanes[-1])
        self.norm_name5 = 'bn5' if not use_gn else 'gn5'
        self.add_module(self.norm_name5,
                        norm_layer(stage_outplanes[-1], use_gn))

        self.feat_dim = stage_outplanes[-1]

    def forward(self, x):
        x = self.conv1(x)
        norm1 = getattr(self, self.norm_name1)
        x = norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, stage_name in enumerate(self.shuffle_stages):
            shuffle_stage = getattr(self, stage_name)
            x = shuffle_stage(x)
            if i in self.out_indices:
                if i < 2:
                    outs.append(x)
                else:
                    x = self.conv5(x)
                    norm5 = getattr(self, self.norm_name5)
                    x = norm5(x)
                    x = self.relu(x)
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
        super(ShuffleNetV2, self).train(mode)
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
            norm1 = getattr(self, self.norm_name1)
            for param in norm1.parameters():
                param.requires_grad = False
            norm1.eval()
            norm1.weight.requires_grad = False
            norm1.bias.requires_grad = False
            for i in range(2, self.frozen_stages + 1):
                if i < 5:
                    mod = getattr(self, 'stage{}'.format(i))
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False
                else:
                    for param in self.conv5.parameters():
                        param.requires_grad = False
                    norm5 = getattr(self, self.norm_name5)
                    for param in norm5.parameters():
                        param.requires_grad = False
                    norm5.eval()
                    norm5.weight.requires_grad = False
                    norm5.bias.requires_grad = False
