"""
Duration Predictor and ONNX-compatible Length Regulator for Indic TTS.
"""

import torch
import torch.nn as nn
from .modules import LayerNorm, sequence_mask

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(p_dropout)

        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(p_dropout)

        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        # x: [B, C, T]
        x = self.conv_1(x * x_mask)
        x = self.norm_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)

        x = self.conv_2(x * x_mask)
        x = self.norm_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)

        x = self.proj(x * x_mask)
        return x * x_mask

def expand_durations(durations, max_len=None):
    """
    ONNX-safe expansion matrix from token durations.
    durations: [B, T_text]
    returns: [B, T_text, T_mel] alignment matrix
    """
    batch_size, text_len = durations.size()
    total_durations = torch.sum(durations, dim=1)
    if max_len is None:
        max_len = total_durations.max().long().item()
        
    cum_dur = torch.cumsum(durations, dim=1)
    cum_prev = cum_dur - durations
    
    # Grid of frames: [1, 1, max_len]
    frames = torch.arange(max_len, device=durations.device).view(1, 1, -1)
    
    # [B, T_text, 1]
    cum_prev_exp = cum_prev.unsqueeze(-1)
    cum_dur_exp = cum_dur.unsqueeze(-1)
    
    # Indicator: frame belongs to token i if cum_prev[i] <= frame < cum_dur[i]
    mask = (frames >= cum_prev_exp) & (frames < cum_dur_exp)
    return mask.float(), total_durations
