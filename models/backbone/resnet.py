import logging
import torch.nn as nn

from ..utils import conv3x3_group, norm_layer, conv1x1_group, \
    kaiming_init, constant_init, load_checkpoint


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3_group(
            inplanes, planes * self.expansion, stride, dilation)
        self.conv2 = conv3x3_group(planes * self.expansion,
                                   planes * self.expansion)

        # we want to load pre-trained models
        # for keep the layer name the same as pre-trained models
        norm_layers = []
        norm_layers.append(norm_layer(planes * self.expansion, use_gn))
        norm_layers.append(norm_layer(planes * self.expansion, use_gn))
        self.norm_names = ['bn1', 'bn2'] if not use_gn else ['gn1', 'gn2']
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

        out = self.conv2(out)
        norm2 = getattr(self, self.norm_names[1])
        out = norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 use_gn=False,
                 downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1_group(
            inplanes, planes, stride=1)
        self.conv2 = conv3x3_group(planes, planes, stride=stride,
                                   dilation=dilation)
        self.conv3 = conv1x1_group(planes, planes * self.expansion,
                                      stride=1)

        # we want to load pre-trained models
        # for keep the layer name the same as pre-trained models
        norm_layers = []
        norm_layers.append(norm_layer(planes, use_gn))
        norm_layers.append(norm_layer(planes, use_gn))
        norm_layers.append(norm_layer(planes * self.expansion, use_gn))
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

        out = self.conv2(out)
        norm2 = getattr(self, self.norm_names[1])
        out = norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        norm3 = getattr(self, self.norm_names[2])
        out = norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def _make_res_layer(block,
                    inplanes,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    use_gn=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1_group(inplanes,
                      planes * block.expansion,
                      stride=stride),
            norm_layer(planes * block.expansion)
        )

    layers = []
    layers.append(
        block(inplanes,
              planes,
              stride=stride,
              dilation=dilation,
              use_gn=use_gn,
              downsample=downsample)
    )
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  planes,
                  stride=1,
                  dilation=dilation,
                  use_gn=use_gn))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
    ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
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
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        assert 1 <= num_stages <= 4
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        if not use_gn:
            self.bn_eval = bn_eval
            self.bn_frozen = bn_frozen
        self.use_gn = use_gn

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm_name = 'bn1' if not use_gn else 'gn1'
        self.add_module(self.norm_name, norm_layer(64, use_gn))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = _make_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                use_gn=use_gn)
            self.inplanes = planes * block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = block.expansion * 64 * 2 ** (len(stage_blocks) - 1)

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
        norm1 = getattr(self, self.norm_name)
        x = norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
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
