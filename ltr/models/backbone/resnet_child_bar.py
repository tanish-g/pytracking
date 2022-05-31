import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from .base import Backbone
import numpy as np
import time
from collections import OrderedDict
import pickle

class MixedBlock(nn.Module):
    def __init__(self, f_delta, delta_idx, in_idx, f_res=None, res_idx=None, res_size=None):
        super(MixedBlock, self).__init__()
        self.f_delta = f_delta
        self.delta_idx = delta_idx
        self.in_idx = in_idx
        self.f_res = f_res
        self.res_idx = res_idx
        self.res_size = res_size
        self.activ = nn.ReLU(inplace=False)  #default true but set to false for checking

        self.res_scatter_idx = None
        self.delta_scatter_idx = None

        if f_delta is None:
            self.forward = self.forward_without_delta
        else:
            self.forward = self.forward_with_delta

    def scatter_features(self, idx, src, final_size):
        idx = idx.cuda()
        if self.res_scatter_idx is None or self.res_scatter_idx.size(0) != src.size(0):
            scatter_idx = idx.new_empty(*src.size()).cuda()
            scatter_idx.copy_(idx[None, :, None, None])
            self.res_scatter_idx = scatter_idx

        x = torch.zeros(src.size(0), final_size, src.size(2), src.size(3)).to(src)
        x.scatter_(dim=1, index=self.res_scatter_idx, src=src)
        return x

    def scatter_add_features(self, dst, idx, src):
        idx = idx.cuda()
        if self.delta_scatter_idx is None or self.delta_scatter_idx.size(0) != src.size(0):
            scatter_idx = idx.new_empty(*src.size()).cuda()
            scatter_idx.copy_(idx[None, :, None, None])
            self.delta_scatter_idx = scatter_idx

        dst.scatter_add_(dim=1, index=self.delta_scatter_idx, src=src)

    def forward_with_delta(self, x):
        # x: (B, C, H, W)

        if self.f_res is None:
            x_alive = x.index_select(dim=1, index=self.in_idx.cuda()).cuda()
            delta = self.f_delta.forward(x_alive).cuda()  # 3x3 conv
        else:
            delta = self.f_delta.forward(x).cuda()  # 3x3 conv

            res = self.f_res.forward(x).cuda()  # 1x1 conv
            x = self.scatter_features(self.res_idx.cuda(), res, self.res_size).cuda()

        self.scatter_add_features(x, self.delta_idx, delta)
        
        return self.activ(x)

    def forward_without_delta(self, x):
        res = self.f_res.forward(x).cuda()  # 1x1 conv
        x = self.scatter_features(self.res_idx.cuda(), res, self.res_size).cuda()

        return self.activ(x)

    @staticmethod
    def from_basic(block, delta_idx, in_idx, res_idx=None, res_size=None):
        f_delta = nn.Sequential(
            block.conv1,
            block.bn1,
            block.activ,  # nn.ReLU(inplace=True)
            block.conv2,
            block.bn2
        )
        return MixedBlock(f_delta, delta_idx, in_idx, block.downsample, res_idx, res_size)
    
    @staticmethod
    def from_bottleneck(block, delta_idx, in_idx, res_idx=None, res_size=None):
        f_delta = nn.Sequential(
            block.conv1,
            block.bn1,
            block.relu,  # nn.ReLU(inplace=True)
            block.conv2,
            block.bn2,
            block.relu,
            block.conv3,
            block.bn3
        )
        return MixedBlock(f_delta, delta_idx, in_idx, block.downsample, res_idx, res_size)

class IdentityModule(nn.Module):
    def forward(self, x):
        return x

with open('/workspace/tracking_datasets/cfg_dict_resnet_child/cfg_dict_resnet50_layerwise_budget_50.json', 'rb') as fp:
    cfg_dict = pickle.load(fp)

def convert_resnet(net, insert_identity_modules=False):
    """Convert a ResNetCifar module (in place)

    Returns
    -------
        net: the mutated net
    """
    
    net.conv1, net.bn1 = convert_conv_bn(net.conv1, net.bn1, torch.ones(3).byte(), (cfg_dict['bn1']>0))
    in_gates = torch.ones(net.conv1.out_channels).byte()

    clean_res = True
    net.layer1, in_gates = convert_layer(net.layer1, in_gates, insert_identity_modules, clean_res, layer_name = 'layer1')
    net.layer2, in_gates = convert_layer(net.layer2, in_gates, insert_identity_modules, clean_res, layer_name = 'layer2')
    net.layer3, in_gates = convert_layer(net.layer3, in_gates, insert_identity_modules, clean_res, layer_name = 'layer3')
    net.layer4, in_gates = convert_layer(net.layer4, in_gates, insert_identity_modules, clean_res, layer_name = 'layer4')


    if clean_res:
        net.fc = convert_fc_head(net.fc, in_gates)
    else:
        net.fc = resnet.InwardPrunedLinear(convert_fc_head(net.fc, in_gates), mask2i(in_gates))

    return net


def convert_layer(layer_module, in_gates, insert_identity_modules, clean_res, layer_name =None):
    """Convert a ResnetCifar layer (in place)

    Parameters
    ----------
        layer_module: a nn.Sequential
        in_gates: mask

    Returns
    -------
        layer_module: mutated layer_module
        in_gates: ajusted mask
    """

    previous_layer_gates = in_gates

    new_blocks = []
    for block_num, block in enumerate(layer_module):
        new_block, in_gates = convert_block(block, in_gates, block_name = layer_name+'.'+str(block_num))
        if new_block is None:
            if insert_identity_modules:
                new_blocks.append(IdentityModule())
        else:
            new_blocks.append(new_block)

    # Remove unused residual features
    if clean_res:
        print()
        cur_layer_gates = in_gates
        for block in new_blocks:
            if isinstance(block, IdentityModule):
                continue
            clean_block(block, previous_layer_gates, cur_layer_gates)  # in-place

    layer_module = nn.Sequential(*new_blocks)
    return layer_module, in_gates


def clean_block(mixed_block, previous_layer_alivef, cur_layer_alivef):
    """Remove unused res features (operates in-place)"""

    def clean_indices(idx, alive_mask=cur_layer_alivef):
        mask = i2mask(idx, alive_mask)
        mask = mask[mask2i(alive_mask)]
        return mask2i(mask)

    if mixed_block.f_res is None:
        mixed_block.in_idx = clean_indices(mixed_block.in_idx)
    else:
        mixed_block.in_idx = clean_indices(mixed_block.in_idx, alive_mask=previous_layer_alivef)
        mixed_block.res_size = cur_layer_alivef.sum().item()
#         print('DOWNS ----- Res size: ', mixed_block.res_size)
        mixed_block.res_idx = clean_indices(mixed_block.res_idx)
#         print('Res:  ', len(mixed_block.res_idx))
    mixed_block.delta_idx = clean_indices(mixed_block.delta_idx)

    print('In:   ', len(mixed_block.in_idx))
    print('Delta:', len(mixed_block.delta_idx))


def convert_block(block_module, in_gates, block_name = None):
    """Convert a Basic Resnet block (in place)

    Parameters
    ----------
        block_module: a BasicBlock
        in_gates: received mask

    Returns
    -------
        block_module: mutated block
        in_gates: out_gates of this block (in_gates for next block)
    """

#     assert not hasattr(block_module, 'conv3')  # must be basic block

    b1_gates = (cfg_dict[f'{block_name}.bn1']>0)   # get_gates(block_module.bn1)
    b2_gates = (cfg_dict[f'{block_name}.bn2']>0)
    b3_gates = (cfg_dict[f'{block_name}.bn3']>0)
   

    delta_branch_is_pruned = b1_gates.sum().item() == 0 or b2_gates.sum().item() == 0 or b3_gates.sum().item() == 0
    
    # Delta branch
    if not delta_branch_is_pruned:
        block_module.conv1, block_module.bn1 = convert_conv_bn(block_module.conv1, block_module.bn1, in_gates, b1_gates)
        block_module.conv2, block_module.bn2 = convert_conv_bn(block_module.conv2, block_module.bn2, b1_gates, b2_gates)
        block_module.conv3, block_module.bn3 = convert_conv_bn(block_module.conv3, block_module.bn3, b2_gates, b3_gates)


    if block_module.downsample is not None:
        ds_gates = (cfg_dict[f'{block_name}.downsample.1']>0) # get_gates(block_module.downsample[1])
        ds_conv, ds_bn = convert_conv_bn(block_module.downsample[0], block_module.downsample[1], in_gates, ds_gates)
        ds_module = nn.Sequential(ds_conv, ds_bn)

        if delta_branch_is_pruned:
            mixed_block = MixedBlock(f_delta=None, delta_idx=None,
                                            f_res=ds_module,
                                            in_idx=mask2i(in_gates),
                                            res_idx=mask2i(ds_gates),
                                            res_size=len(b3_gates))
        else:
            block_module.downsample = ds_module
            mixed_block = MixedBlock.from_bottleneck(block_module,
                                                       delta_idx=mask2i(b3_gates),
                                                       in_idx=mask2i(in_gates),
                                                       res_idx=mask2i(ds_gates),
                                                       res_size=len(b3_gates))
        in_gates = elementwise_or(ds_gates, b3_gates)
    else:
        if delta_branch_is_pruned:
            mixed_block = None
        else:
            mixed_block = MixedBlock.from_bottleneck(block_module,
                                                       delta_idx=mask2i(b3_gates),
                                                       in_idx=mask2i(in_gates))
        in_gates = elementwise_or(in_gates, b3_gates)

    return mixed_block, in_gates


def convert_conv_bn(conv_module, bn_module, in_gates, out_gates):
    in_indices = mask2i(in_gates)  # indices of kept features
    out_indices = mask2i(out_gates)

    # Keep the good ones
    new_conv_w = conv_module.weight.data[out_indices][:, in_indices]

    new_conv = make_conv(new_conv_w, from_module=conv_module)
    new_bn = convert_bn(bn_module, out_indices)

    new_conv.out_idx = out_indices
    
    return new_conv, new_bn


def convert_fc_head(fc_module, in_gates):
    """Convert a the final FC module of the net

    Parameters
    ----------
        fc_module: a nn.Linear with weight tensor of size (out_f, in_f)
        in_gates: binary vector or list of size in_f

    Returns
    -------
        fc_module: mutated module
    """

    in_indices = mask2i(in_gates)
    new_weight_tensor = fc_module.weight.data[:, in_indices]
    return make_fc(new_weight_tensor, from_module=fc_module)


def convert_bn(bn_module, out_indices):
#     z = bn_module.get_gates(stochastic=False)
    new_weight = bn_module.weight.data[out_indices] # * z[out_indices]
    new_bias = bn_module.bias.data[out_indices] # * z[out_indices]

    new_bn_module = nn.BatchNorm2d(len(new_weight))
    new_bn_module.weight.data.copy_(new_weight)
    new_bn_module.bias.data.copy_(new_bias)
    new_bn_module.running_mean.copy_(bn_module.running_mean[out_indices])
    new_bn_module.running_var.copy_(bn_module.running_var[out_indices])

    new_bn_module.out_idx = out_indices

    return new_bn_module


def make_bn(bn_module, kept_indices):
    new_bn_module = nn.BatchNorm2d(len(kept_indices))
    new_bn_module.weight.data.copy_(bn_module.weight.data[kept_indices])
    new_bn_module.bias.data.copy_(bn_module.bias.data[kept_indices])
    new_bn_module.running_mean.copy_(bn_module.running_mean[kept_indices])
    new_bn_module.running_var.copy_(bn_module.running_var[kept_indices])

    if hasattr(bn_module, 'out_idx'):
        new_bn_module.out_idx = bn_module.out_idx[kept_indices]
    else:
        new_bn_module.out_idx = kept_indices

    return new_bn_module


def make_conv(weight_tensor, from_module):
    # NOTE: No bias

    # New weight size
    in_channels = weight_tensor.size(1)
    out_channels = weight_tensor.size(0)

    # Other params
    kernel_size = from_module.kernel_size
    stride = from_module.stride
    padding = from_module.padding

    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    conv.weight.data.copy_(weight_tensor)
    return conv


def make_fc(weight_tensor, from_module):
    in_features = weight_tensor.size(1)
    out_features = weight_tensor.size(0)
    fc = nn.Linear(in_features, out_features)
    fc.weight.data.copy_(weight_tensor)
    fc.bias.data.copy_(from_module.bias.data)
    return fc

def elementwise_or(a, b):
    return (a + b) > 0


def mask2i(mask):
#     assert mask.dtype == torch.uint8
    return mask.nonzero().view(-1)  # Note: do not use .squeeze() because single item becomes a scalar instead of 1-vec


def i2mask(i, from_tensor):
    x = torch.zeros_like(from_tensor)
    x[i] = 1
    return x

    
    
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

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
        self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = self.bn1(x)
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


def resnet50_bar(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    model = convert_resnet(model)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101_bar(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck,[3, 4, 23, 3], output_layers, **kwargs)
    model = convert_resnet(model)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

