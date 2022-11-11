# DeepGOld
# Reconfigure a pretrained network
# copyright 2022 moshe sipper  
# www.moshesipper.com 

import torch.nn as nn
import torchvision.models as vismodels
from datasets import Datasets

def get_pretrained_names(): # https://pytorch.org/vision/stable/models.html
    names = sorted(name for name in vismodels.__dict__ if name.islower() and not name.startswith("__") and callable(vismodels.__dict__[name]))     
    FurtherTreatment = ['inception_v3', 'mnasnet0_75', 'mnasnet1_3', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'googlenet'] # These networks require additional processing or are not supported, and are not used for now
    for network in FurtherTreatment: 
        names.remove(network)
    return names
'''
Pretrained = get_pretrained_names() # run once, save list below
'''
Pretrained = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'mnasnet0_5', 'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']

    
def _remove_grad(model):
    for param in model.parameters(): 
        param.requires_grad = False

def _closest_power_of_2(n):
    res = 1
    while res <= n:
        prev = res
        res *= 2
    return prev

def _fc(num_ftrs , n_classes, fc_type='multiple'):
    if fc_type == 'multiple':
        inp = num_ftrs
        out = _closest_power_of_2(inp)
        layers= []
        while out > n_classes:
            layers += [nn.Linear(inp, out),
                      nn.BatchNorm1d(out),
                      nn.LeakyReLU()]
            inp = out
            out = int(out/2)
        layers += [nn.Linear(inp, n_classes),
                  nn.BatchNorm1d(n_classes),
                  nn.LeakyReLU()]
          
        fc  = nn.Sequential(*layers) 
        return fc        
    
    elif fc_type == 'single':
        return nn.Linear(in_features=num_ftrs, out_features=n_classes, bias=True)    
        
    else: 
        exit(f'Error: unknown fc_type {fc_type}')

def _replace_layer(network, new, n_classes, fc_type, done, first=True):
    '''    
    replace either first or last layer in `network' to `new'
    assume first layer is Conv2d and last layer is Linear
    done is a size-1 list to induce call by reference in recursive calls
    based on: https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
        also: https://www.kaggle.com/ankursingh12/why-use-setattr-to-replace-pytorch-layers
        also: https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7  
    '''
    if first:  
        iter = network._modules.items() # network.named_children()
    else:
        iter = reversed(network._modules.items())
    for name, module in iter:
        if len(list(module.children())) > 0: # recurse
            _replace_layer(module, new, n_classes, fc_type, done, first)
        elif not done[0]:
            if first:
                network._modules[name] = new
            else:
                network._modules[name] = _fc(network._modules[name].in_features, n_classes, fc_type) 
            done[0] = True
            return
        else:
            return

def _modify_network(network, first, last, n_classes, fc_type, remove_grad):
    # replace first and last layers, possibly remove grads
    _replace_layer(network, first, n_classes, fc_type, done=[False], first=True)
    if remove_grad: _remove_grad(network) # after first, before last, so that grads of last layer remain
    _replace_layer(network, last, n_classes, fc_type, done=[False], first=False)
    
class Net(nn.Module):
    def __init__(self, net_name, dataset, fc_type, remove_grad):
        assert net_name in Pretrained, f'Error: Given net_name={net_name}, must by one of {Pretrained}.'
        super(Net, self).__init__()
        
        n_classes = Datasets[dataset]['n_classes']
        input_shape = Datasets[dataset]['input_shape']
        in_channels = input_shape[1]
        
        out_channels = 64
        if net_name in ['mnasnet0_5', 'mobilenet_v3_large', 'mobilenet_v3_small']:
            out_channels = 16
        elif net_name in ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']:
            out_channels = 24
        elif net_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'inception_v3', 'mnasnet1_0', 'mobilenet_v2', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf']:
            out_channels = 32
        elif net_name in ['efficientnet_b3']:
            out_channels = 40
        elif net_name in ['efficientnet_b4', 'efficientnet_b5']:
            out_channels = 48
        elif net_name in ['efficientnet_b6']:
            out_channels = 56
        elif net_name in ['densenet161']:
            out_channels = 96
        
        network = vismodels.__dict__[net_name](pretrained=True)
        first = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if 'vgg' in net_name:
            first = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif net_name in ['inception_v3']:
            first = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        last = None # _replace_layer will figure this out...
        _modify_network(network, first, last, n_classes, fc_type, remove_grad)
        self.main = network
                        
    def forward(self, x):
        return self.main(x)

# end class Net
