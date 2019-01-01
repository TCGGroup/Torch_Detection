import torch.nn as nn
import torch.nn.functional as F

from ..utils import ConvModule, xavier_init, constant_init


class PAFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 use_gn=False,
                 activation=None):
        super(PAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_bias = normalize is None
        self.activation = activation

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.backbone_end_level - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= self.num_ins
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pa_convs1 = nn.ModuleList()
        self.pa_convs2 = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(self.in_channels[i],
                                self.out_channels,
                                kernel_size=1,
                                bias=self.with_bias,
                                normalize=normalize,
                                use_gn=use_gn)
            fpn_conv = ConvModule(self.out_channels,
                                  self.out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=self.with_bias,
                                  normalize=normalize,
                                  use_gn=use_gn)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            if i < self.backbone_end_level - 1:
                pa_conv1 = ConvModule(self.out_channels,
                                      self.out_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=self.with_bias,
                                      normalize=normalize,
                                      use_gn=use_gn,
                                      activation=self.activation)
                pa_conv2 = ConvModule(self.out_channels,
                                      self.out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      bias=self.with_bias,
                                      normalize=normalize,
                                      use_gn=use_gn,
                                      activation=self.activation)
                self.pa_convs1.append(pa_conv1)
                self.pa_convs2.append(pa_conv2)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channel = self.in_channels[self.backbone_end_level - 1] \
                    if i == 0 else self.out_channels
                extra_fpn_conv = ConvModule(
                    in_channel,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=self.with_bias,
                    normalize=normalize,
                    use_gn=use_gn)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        use_backbone_levels = len(laterals)
        for i in range(use_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build smooth outputs, P
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(use_backbone_levels)]

        # build bottom-up path (PA path), N
        # index begin with 1, because the coarsest level N is same as in P
        for i in range(1, use_backbone_levels):
            outs[i] = self.pa_convs2[i - 1](
                outs[i] + self.pa_convs1[i - 1](outs[i - 1]))

        # add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - use_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[use_backbone_levels](orig))
                for i in range(use_backbone_levels + 1, self.num_outs):
                    # P6 --> ReLU --> P7
                    outs.append(
                        self.fpn_convs[i](F.relu(outs[-1], inplace=True)))
        return tuple(outs)
