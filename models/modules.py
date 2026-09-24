"""
Core neural modules for compact TTS student architecture.
Optimized for ONNX export, minimal memory, and fast CPU inference.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class LayerNorm(nn.Module):
    """Channel-first 1D Layer Normalization for ONNX compatibility."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x):
        # x: [B, C, T]
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

class WN(nn.Module):
    """Weight Normalization ConvNet for Flow & Duration Predictors."""
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = get_padding(kernel_size, dilation)
            in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
            self.in_layers.append(weight_norm(in_layer))

            if i < n_layers - 1:
                res_skip_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            else:
                res_skip_layer = nn.Conv1d(hidden_channels, hidden_channels, 1)
            self.res_skip_layers.append(weight_norm(res_skip_layer))

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.tensor(self.hidden_channels)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            acts = F.glu(x_in, dim=1)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
