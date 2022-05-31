import math
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
import numpy as np
import torch

class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()
    
class Backbone(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """
    def __init__(self, frozen_layers=()):
        super().__init__()

        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError('Unknown option for frozen layers: \"{}\". Should be \"all\", \"none\" or list of layer names.'.format(frozen_layers))

        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False


    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True
        return self


    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()


    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)
                    
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class EcaModule(nn.Module):
    """Constructs an ECA module.
    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """
    def __init__(
            self, channels=None, kernel_size=5, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid',
            rd_ratio=1/8, rd_channels=None, rd_divisor=8, use_mlp=False):
        super(EcaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        if use_mlp:
            # NOTE 'mlp' mode is a timm experiment, not in paper
            assert channels is not None
            if rd_channels is None:
                rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
            act_layer = act_layer or nn.ReLU
            self.conv = nn.Conv1d(1, rd_channels, kernel_size=1, padding=0, bias=True)
            self.act = create_act_layer(act_layer)
            self.conv2 = nn.Conv1d(rd_channels, 1, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.act = None
            self.conv2 = None
        self.gate = Sigmoid()

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        y = self.conv(y)
        if self.conv2 is not None:
            y = self.act(y)
            y = self.conv2(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, dilation=1,):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0],cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[3])
        self.se = EcaModule(None,kernel_size=5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
#         if self.downsample is not None:
#             self.downsample[0].in_channels = cfg[0]
#             self.downsample[0].out_channels = cfg[3]
#             self.downsample[1].num_features = cfg[3]
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)
#         out = self.select(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers,cfg = None , num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes , kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1), cfg = cfg[0:3*layers[0]+1])
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1), cfg = cfg[3*layers[0]:3*layers[1]+3*layers[0]+1])
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1), cfg = cfg[3*layers[1]+3*layers[0]:3*layers[1]+3*layers[0]+3*layers[2]+1])
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor, cfg = cfg[3*layers[1]+3*layers[0]+3*layers[2]:3*layers[1]+3*layers[0]+3*layers[2]+3*layers[3]+1])

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, cfg=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Identity(),
                nn.Conv2d(cfg[0], cfg[3],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cfg[3]),
#                 channel_selection(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:4], stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
#             if i==blocks-1 and self.inplanes==1024:
#                 print('yes')
#                 bef = cfg[3*(i+1)]
#                 cfg[3*(i+1)] = 1024
#                 layers.append(block(self.inplanes, planes, cfg[(3*i):(3*(i+1))+1]))
#                 cfg[3*(i+1)] = bef
#             else:
             layers.append(block(self.inplanes, planes, cfg[(3*i):(3*(i+1))+1]))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x) ####### select daal dena
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=inplanes, **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


def resnet18(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50_child(output_layers=None, pretrained=False,cfg = None,**kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
#     ckpt = torch.load('/workspace/tracking_datasets/pruned_ckpts/dimp50_correct/dimp50_correct.pth.tar')
#     cfg = ckpt['cfg']
    model = ResNet(Bottleneck, [3, 4, 6, 3],output_layers,cfg=cfg,**kwargs)
#     if pretrained:
# #         model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], progress = False), strict = False )
#           model.load_state_dict(ckpt['state_dict'])
#           print('pruned checkpoint loaded')
    return model

def resnet101_child(output_layers=None, pretrained=False,cfg = None,**kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
#     ckpt = torch.load('/workspace/tracking_datasets/pruned_ckpts/dimp50_correct/dimp50_correct.pth.tar')
#     cfg = ckpt['cfg']
    model = ResNet(Bottleneck,[3, 4, 23, 3], output_layers,cfg=cfg,**kwargs)
#     if pretrained:
# #         model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], progress = False), strict = False )
#           model.load_state_dict(ckpt['state_dict'])
#           print('pruned checkpoint loaded')
    return model

def ecaresnet50d(output_layers=None, pretrained=False,cfg = None,**kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
#     ckpt = torch.load('/workspace/tracking_datasets/pruned_ckpts/dimp50_correct/dimp50_correct.pth.tar')
#     cfg = ckpt['cfg']
    cfg = np.load('/workspace/ecaresnet.npy')
    model = ResNet(Bottleneck, [3, 4, 6, 3],output_layers,cfg=cfg,**kwargs)
    if pretrained:
        ckpt = torch.load('/workspace/ecaresnet50d_pruned.pth')
        model.load_state_dict(ckpt,strict=False)
#     if pretrained:
# #         model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], progress = False), strict = False )
#           model.load_state_dict(ckpt['state_dict'])
#           print('pruned checkpoint loaded')
    return model