"""
Exportable ONNX wrapper for Student VITS.
Self-contained, zero training baggage, fully dynamic input/output shapes.
Compatible with Android ONNX Runtime, onnxruntime-node, and onnxruntime-web.
"""

import torch
import torch.nn as nn
from .modules import sequence_mask
from .duration_predictor import expand_durations

class ExportableVITS(nn.Module):
    def __init__(self, student_model):
        super().__init__()
        self.enc_p = student_model.enc_p
        self.flow = student_model.flow
        self.dp = student_model.dp
        self.dec = student_model.dec

        # Remove weight normalization for optimized inference
        try:
            self.flow.remove_weight_norm()
        except Exception:
            pass
        try:
            self.dec.remove_weight_norm()
        except Exception:
            pass

    def forward(self, input_ids: torch.Tensor, input_lengths: torch.Tensor, scales: torch.Tensor):
        """
        Args:
            input_ids: [B, T_text] int64
            input_lengths: [B] int64
            scales: [3] float32 (noise_scale, length_scale, noise_scale_w)
        Returns:
            audio: [B, 1, T_audio] float32
        """
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        # 1. Encode text
        x_feat, m_p, logs_p, x_mask = self.enc_p(input_ids, input_lengths)

        # 2. Predict durations
        logw = self.dp(x_feat, x_mask)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w).squeeze(1).long().clamp(min=1)

        # 3. Dynamic expansion along duration
        # Matrix multiply [B, C, T_x] x [B, T_x, T_mel] -> [B, C, T_mel]
        align_mask, total_durations = expand_durations(w_ceil)
        m_p_exp = torch.bmm(m_p, align_mask)
        logs_p_exp = torch.bmm(logs_p, align_mask)

        max_mel_len = m_p_exp.size(2)
        y_mask = sequence_mask(total_durations, max_mel_len).unsqueeze(1).float()

        # 4. Latent prior sampling
        # In deterministic ONNX runtime on CPU, setting noise_scale=0 or small constant gives clean, deterministic speech
        eps = torch.randn_like(m_p_exp)
        z_p = (m_p_exp + eps * torch.exp(logs_p_exp) * noise_scale) * y_mask

        # 5. Invert normalizing flow: z_p -> z
        z = self.flow(z_p, y_mask, reverse=True)

        # 6. Synthesize audio waveform with HiFi-GAN
        audio = self.dec(z * y_mask)

        return audio
