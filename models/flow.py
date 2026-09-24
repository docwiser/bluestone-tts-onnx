"""
Residual Coupling Layers (Normalizing Flow) for VITS.
Invertible transformation for mapping latent distributions.
"""

import torch
import torch.nn as nn
from .modules import WN

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0.0, gin_channels=0):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        m = self.post(h) * x_mask

        if not reverse:
            x1 = (x1 + m) * x_mask
            return torch.cat([x0, x1], dim=1)
        else:
            x1 = (x1 - m) * x_mask
            return torch.cat([x0, x1], dim=1)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
                # Flip channels along dim=1
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=True)
        return x

    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()
