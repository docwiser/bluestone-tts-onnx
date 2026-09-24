"""
Lightweight Text Encoder for Indic TTS.
Optimized for ONNX export and real-time CPU evaluation.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules import LayerNorm, sequence_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, c, attn_mask=None):
        # x: [B, C, T]
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        b, c, t_t = q.size()
        t_s = k.size(2)

        q = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, v)
        output = output.transpose(2, 3).contiguous().view(b, self.channels, t_t)
        return self.conv_o(output)

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        h = self.conv_1(x * x_mask)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.conv_2(h * x_mask)
        return h * x_mask

class EncoderLayer(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size=3, p_dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
        self.norm1 = LayerNorm(hidden_channels)
        self.ffn = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.norm2 = LayerNorm(hidden_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # Self-attention
        attn_out = self.attn(x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward
        ffn_out = self.ffn(x, x_mask)
        x = self.norm2(x + self.drop(ffn_out))
        return x * x_mask

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, out_channels, hidden_channels, filter_channels,
                 n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(vocab_size, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.prenet = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

        # Projection to mu and log(sigma)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        # x: [B, T_text]
        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1).to(x.dtype)
        
        # Embedding: [B, T, C] -> [B, C, T]
        x = self.emb(x).transpose(1, 2) * math.sqrt(self.hidden_channels)
        x = self.prenet(x * x_mask)

        for layer in self.layers:
            x = layer(x, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
