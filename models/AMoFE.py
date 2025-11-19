import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from einops import rearrange
import scipy.linalg
from . import thops
# import thops

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1, padding = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding = padding, bias=bias, stride = stride)


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super(ResBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, (in_channels+s_factor)//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, (in_channels+s_factor) * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):    
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels+s_factor, in_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x       

class RSM(nn.Module):
    '''For residual supervision'''
    def __init__(self, n_feat, n2_feat, kernel_size, bias):
        super(RSM, self).__init__()
        self.conv_pred= nn.Conv2d(n_feat, n2_feat, kernel_size=1)  # for residual supervision
        self.conv_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n2_feat, n_feat, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, I_up):
        rs = self.conv_pred(x)
        x_sup = rs + I_up                     
        att = self.conv_attn(rs)             
        x = x * att + x                   
        return x, x_sup
    
# MoE: gate + experts
class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k, noise_strength=0.1):
        super(GateNetwork, self).__init__()

        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_strength = noise_strength

        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)

        self.fc0 = nn.Linear(input_size, num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()


    def forward(self, x):
        x = self.pool_max(x)+self.pool_avg(x)
        x = x.view(-1, self.input_size)
        inp = x
        x = self.fc0(x)
        x= self.act(x)

        noise = self.sp(self.fc1(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std

        _ , topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')
        gate = self.softmax(x)      # [B, E]

        return gate

class SpatialExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, hyper, multi):
        x = torch.cat([hyper, multi], dim=1)
        x = F.leaky_relu(self.conv1(x))
        
        attn = self.spatial_attention(x)
        x = x * attn
        
        x = self.conv2(x)
        return x

class SpectralExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hyper_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.multi_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.fusion = nn.Conv2d(channels*2, channels, kernel_size=1)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, hyper, multi):
        h = self.hyper_proj(hyper)
        m = self.multi_proj(multi)
        
        h_attn = self.channel_attention(h)
        h = h * h_attn
        
        x = torch.cat([h, m], dim=1)
        x = self.fusion(x)
        return x

class EdgeExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_filters = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        for i in range(channels):
            self.edge_filters.weight.data[i, 0] = torch.FloatTensor(kernel).unsqueeze(0)
        
        self.conv1 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, hyper, multi):
        hyper_edge = torch.abs(self.edge_filters(hyper))
        multi_edge = torch.abs(self.edge_filters(multi))
        
        edge_fusion = torch.cat([hyper_edge, multi_edge], dim=1)
        edge_fusion = F.leaky_relu(self.conv1(edge_fusion))
        edge_fusion = self.conv2(edge_fusion)
        
        return edge_fusion

class AMoFE(nn.Module):
    """attribute-aware mixture of fusion experts"""
    def __init__(self,channels, num_experts=3, k=3):
        super(AMoFE, self).__init__()
        self.gate = GateNetwork(channels, num_experts, k)
 
        self.expert_networks_d = nn.ModuleList([
            SpatialExpert(channels),     # Expert 0: Spatial expert
            SpectralExpert(channels),    # Expert 1: Spectral expert
            EdgeExpert(channels),        # Expert 2: Edge Expert
        ])

        self.pre_fuse = nn.Conv2d(2*channels, channels, 1, 1, 0)
        self.num_experts = num_experts

    def forward(self, x, y):
        x_ = self.pre_fuse(torch.cat([x, y], dim=1))
        cof = self.gate(x_)
        out = torch.zeros_like(x_).to(x_.device)

        for idx in range(self.num_experts):
            expert_out = self.expert_networks_d[idx](x, y)  
            coef = cof[:, idx].view(-1, 1, 1, 1) 
            out += expert_out * coef

        return out, cof
