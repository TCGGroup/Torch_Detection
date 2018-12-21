import torch.nn as nn


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


def norm_layer(planes, use_gn=False):
    if not use_gn:
        return nn.BatchNorm2d(planes)
    else:
        return nn.GroupNorm(32, planes)
