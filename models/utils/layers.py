import torch.nn as nn


def conv1x1_group(in_planes, out_planes, stride=1, groups=1):
    """
    1x1 convolution with group, without bias
    - Normal 1x1 convolution when groups == 1
    - Grouped 1x1 convolution when groups > 1
    """
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=1,
                     stride=stride,
                     groups=groups,
                     bias=False)


def conv3x3_group(in_planes, out_planes, stride=1, dilation=1, groups=1):
    """
    3x3 convolution with padding and group, without bias, in this situation,
    padding is same as dilation.
    """
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     dilation=dilation,
                     groups=groups,
                     bias=False)


def conv7x7_group(in_planes, out_planes, stride=1, groups=1):
    """
    7x7 convolution with padding and group, without bias, as first conv
    dilation is set to 1 and padding set to 3.
    """
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     dilation=1,
                     groups=groups,
                     bias=False)


def norm_layer(planes, use_gn=False):
    if not use_gn:
        return nn.BatchNorm2d(planes)
    else:
        return nn.GroupNorm(32, planes)


def get_group_gn(planes):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = -1
    num_groups = 32

    assert dim_per_gp == -1 or num_groups  == -1, \
        'GroupNorm: can only specify G or C/G'

    if dim_per_gp > 0:
        assert planes % dim_per_gp == 0
        groups = planes // dim_per_gp
    else:
        assert planes % num_groups == 0
        groups = num_groups
    return groups


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


class ChannelSplit(nn.Module):
    def __init__(self):
        super(ChannelSplit, self).__init__()

    def forward(self, x):
        half_channel = x.shape[2] // 2
        return x[:, :half_channel, ...], x[:, half_channel:, ...]


class SELayer(nn.Module):
    """
    Paper: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        return x * y
