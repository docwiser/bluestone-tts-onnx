"""
Loss functions for TTS Knowledge Distillation and Adversarial Training.
Includes: Mel L1 Loss, KL Divergence, Discriminator/Generator GAN Losses, Feature Matching Loss.
"""

import torch
import torch.nn.functional as F

def feature_loss(fmap_r, fmap_g):
    """Feature matching loss between discriminator activations."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2.0

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Least-squares GAN loss for Discriminator."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """Least-squares GAN loss for Generator."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        loss += l
        gen_losses.append(l.item())
    return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    KL Divergence between approximate posterior q(z|y) and prior p(z|x).
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss

def mel_spectrogram_loss(mel_real, mel_pred):
    """L1 Loss on mel-spectrograms."""
    return F.l1_loss(mel_pred, mel_real)
