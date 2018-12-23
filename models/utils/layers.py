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


def conv7x7_group(in_planes, out_planes, stride=1, group=1):
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
