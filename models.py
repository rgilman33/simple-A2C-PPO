# Impala CNN. Same as that used in procgen papers except we've added batchnorm. Each module tested separately during construction.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)
    
    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out + x
    
class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x
    
class ImpalaCNN(nn.Module):
    def __init__(self):
        super(ImpalaCNN, self).__init__()
        self.block1 = ImpalaBlock(in_channels=3, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(800, 256)
        
        self.critic = init_critic_(nn.Linear(256, 1))
        self.actor = init_actor_(nn.Linear(256, n_actions))
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        
        c = self.critic(x)
        a = nn.LogSoftmax(dim=-1)(self.actor(x))
        return a, c
    
# Proper orthogonal init in the right locations is important

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
init_critic_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
init_actor_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)