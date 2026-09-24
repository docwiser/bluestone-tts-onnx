"""
Unified Compact VITS Student Model for Indic TTS.
Supports training with knowledge distillation losses and full inference.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .text_encoder import TextEncoder
from .duration_predictor import DurationPredictor, expand_durations
from .flow import ResidualCouplingBlock
from .generator import Generator
from .modules import sequence_mask, WN

def rand_slice_segments(x, x_lengths=None, segment_size=8192):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(b):
        ret[i] = x[i, :, ids_str[i]:ids_str[i] + segment_size]
    return ret, ids_str

def slice_segments(x, ids_str, segment_size=8192):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        ret[i] = x[i, :, ids_str[i]:ids_str[i] + segment_size]
    return ret

class PosteriorEncoder(nn.Module):
    """Posterior encoder used solely during training."""
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class StudentVITS(nn.Module):
    def __init__(self, vocab_size, spec_channels, cfg):
        super().__init__()
        self.vocab_size = vocab_size
        self.spec_channels = spec_channels
        self.cfg = cfg
        
        m_cfg = cfg["model"]
        self.inter_channels = m_cfg["inter_channels"]
        self.hidden_channels = m_cfg["hidden_channels"]
        self.filter_channels = m_cfg["filter_channels"]
        self.n_heads = m_cfg["n_heads"]
        self.n_layers = m_cfg["n_layers"]
        self.kernel_size = m_cfg["kernel_size"]
        self.p_dropout = m_cfg["p_dropout"]
        self.gin_channels = m_cfg.get("gin_channels", 0)

        # 1. Text Encoder
        self.enc_p = TextEncoder(
            vocab_size=vocab_size,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout
        )

        # 2. Posterior Encoder (Training only)
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=self.gin_channels
        )

        # 3. Normalizing Flow
        self.flow = ResidualCouplingBlock(
            channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=self.gin_channels
        )

        # 4. Duration Predictor
        self.dp = DurationPredictor(
            in_channels=self.hidden_channels,
            filter_channels=self.filter_channels // 2,
            kernel_size=3,
            p_dropout=self.p_dropout
        )

        # 5. Generator (HiFi-GAN)
        self.dec = Generator(
            initial_channel=self.inter_channels,
            resblock_kernel_sizes=m_cfg["resblock_kernel_sizes"],
            resblock_dilation_sizes=m_cfg["resblock_dilation_sizes"],
            upsample_rates=m_cfg["upsample_rates"],
            upsample_initial_channel=m_cfg["upsample_initial_channel"],
            upsample_kernel_sizes=m_cfg["upsample_kernel_sizes"],
            gin_channels=self.gin_channels
        )

    def forward(self, x, x_lengths, y, y_lengths, durations=None):
        """Training forward pass."""
        # 1. Text encoding: x_feat: [B, C, T_x], m_p: [B, C, T_x], logs_p: [B, C, T_x]
        x_feat, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # 2. Posterior encoding from audio spectrogram
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)

        # 3. Flow transform: z -> z_p
        z_p = self.flow(z, y_mask)

        # 4. Duration prediction & alignment
        logw_ = self.dp(x_feat, x_mask)

        # If external/teacher durations provided, use them
        if durations is not None:
            w = durations.unsqueeze(1)
            logw = torch.log(w + 1e-6) * x_mask
            loss_dur = torch.sum((logw_ - logw) ** 2) / torch.sum(x_mask)
        else:
            loss_dur = torch.tensor(0.0, device=x.device)

        # 5. Random slice segment for waveform generator
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, segment_size=self.cfg["training"]["segment_size"] // 256)
        o = self.dec(z_slice)

        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), loss_dur

    def infer(self, x, x_lengths, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8, max_len=None):
        """Inference forward pass."""
        x_feat, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Predict durations
        logw = self.dp(x_feat, x_mask)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w).squeeze(1).long().clamp(min=1)

        # Expand along duration
        align_mask, total_durations = expand_durations(w_ceil, max_len=max_len)
        m_p_exp = torch.bmm(m_p, align_mask)
        logs_p_exp = torch.bmm(logs_p, align_mask)
        y_mask = sequence_mask(total_durations, m_p_exp.size(2)).unsqueeze(1).float()

        # Sample z_p from prior
        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale
        z_p = z_p * y_mask

        # Invert flow: z_p -> z
        z = self.flow(z_p, y_mask, reverse=True)

        # Decode z to audio
        o = self.dec(z * y_mask)
        return o, y_mask, (z, z_p, m_p_exp, logs_p_exp)
