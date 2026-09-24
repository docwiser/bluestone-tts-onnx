"""
Student TTS Distillation Training Script.
Supports FP16 mixed-precision on Google Colab GPU and CPU fallback.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import soundfile as sf
from tqdm import tqdm

from models.student_vits import StudentVITS
from models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from frontend.indic_tokenizer import IndicTokenizer
from training.dataset import DistilledTTSDataset, collate_fn, MelSpectrogramComputer
from training.losses import feature_loss, discriminator_loss, generator_loss, kl_loss, mel_spectrogram_loss

def train(config_path: str, manifest_path: str, output_dir: str, epochs: int = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Frontend & Tokenizer
    tokenizer = IndicTokenizer(lang=cfg["language"])
    vocab_size = tokenizer.vocab_size
    print(f"[*] Vocabulary size: {vocab_size}")

    # 2. Dataset & DataLoader
    dataset = DistilledTTSDataset(manifest_path, tokenizer, cfg["audio"])
    batch_size = cfg["training"]["batch_size"]
    # Adjust batch size for CPU or small memory
    if device.type == "cpu" or len(dataset) < batch_size:
        batch_size = max(1, min(4, len(dataset)))
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 3. Models
    spec_channels = cfg["audio"]["filter_length"] // 2 + 1
    net_g = StudentVITS(vocab_size, spec_channels, cfg).to(device)
    net_d = MultiPeriodDiscriminator(use_spectral_norm=cfg["model"]["use_spectral_norm"]).to(device)

    # 4. Optimizers
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=cfg["training"]["betas"],
        eps=cfg["training"]["eps"]
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=cfg["training"]["betas"],
        eps=cfg["training"]["eps"]
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg["training"]["lr_decay"])
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg["training"]["lr_decay"])

    scaler = GradScaler(enabled=(device.type == "cuda" and cfg["training"].get("fp16_run", True)))
    mel_computer = MelSpectrogramComputer(
        sample_rate=cfg["audio"]["sampling_rate"],
        n_fft=cfg["audio"]["filter_length"],
        win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mel_channels"],
        f_min=cfg["audio"]["mel_fmin"],
        f_max=cfg["audio"]["mel_fmax"]
    )

    total_epochs = epochs if epochs is not None else cfg["training"]["epochs"]
    print(f"[*] Starting training for {total_epochs} epochs with {len(dataset)} samples...")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        net_g.train()
        net_d.train()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
        for batch in pbar:
            global_step += 1

            x = batch["text"].to(device)
            x_lengths = batch["text_lengths"].to(device)
            spec = batch["spec"].to(device)
            spec_lengths = batch["spec_lengths"].to(device)
            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device)
            durations = batch["durations"].to(device) if batch["durations"] is not None else None

            # ----------------------
            # 1. Train Discriminator
            # ----------------------
            optim_d.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                y_hat, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), loss_dur = net_g(
                    x, x_lengths, spec, spec_lengths, durations=durations
                )

                # Segment real audio to match sliced generated audio
                segment_size = cfg["training"]["segment_size"]
                y_sliced = torch.zeros(wav.size(0), 1, segment_size, device=device)
                for b_idx in range(wav.size(0)):
                    str_idx = ids_slice[b_idx] * 256
                    end_idx = min(str_idx + segment_size, wav.size(2))
                    actual_len = end_idx - str_idx
                    y_sliced[b_idx, :, :actual_len] = wav[b_idx, :, str_idx:end_idx]

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())
                loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            scaler.scale(loss_d).backward()
            scaler.step(optim_d)

            # ------------------
            # 2. Train Generator
            # ------------------
            optim_g.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
                
                # GAN Loss
                loss_g, _ = generator_loss(y_d_hat_g)
                # Feature Matching Loss
                loss_fm = feature_loss(fmap_r, fmap_g)
                # KL Loss
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

                # Mel-spectrogram loss on generated slice
                mel_slice_pred = mel_computer(y_hat.squeeze(1))
                mel_slice_real = mel_computer(y_sliced.squeeze(1))
                loss_mel = mel_spectrogram_loss(mel_slice_real, mel_slice_pred)

                # Total generator loss with config weights
                c_mel = cfg["training"].get("c_mel", 45.0)
                c_kl = cfg["training"].get("c_kl", 1.0)
                c_fm = cfg["training"].get("c_fm", 2.0)
                c_dur = cfg["training"].get("c_dur", 1.0)

                loss_total_g = loss_g + (loss_fm * c_fm) + (loss_mel * c_mel) + (loss_kl * c_kl) + (loss_dur * c_dur)

            scaler.scale(loss_total_g).backward()
            scaler.step(optim_g)
            scaler.update()

            epoch_loss_g += loss_total_g.item()
            epoch_loss_d += loss_d.item()

            pbar.set_postfix({
                "loss_g": f"{loss_total_g.item():.3f}",
                "loss_mel": f"{loss_mel.item():.3f}",
                "loss_d": f"{loss_d.item():.3f}"
            })

        scheduler_g.step()
        scheduler_d.step()

        # Checkpoint saving
        if epoch % cfg["training"].get("save_interval", 10) == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"student_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": net_g.state_dict(),
                "vocab_size": vocab_size,
                "config": cfg
            }, ckpt_path)
            print(f"[✓] Saved checkpoint: {ckpt_path}")

        # Save best model
        avg_loss = epoch_loss_g / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(checkpoint_dir, "best_student.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": net_g.state_dict(),
                "vocab_size": vocab_size,
                "config": cfg
            }, best_ckpt)

    print(f"\n[✓] Training complete! Best checkpoint saved at: {os.path.join(checkpoint_dir, 'best_student.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student TTS via Distillation")
    parser.add_argument("--config", type=str, default="configs/config_hi.json", help="Path to config file")
    parser.add_argument("--manifest", type=str, default="data/distilled_hi/manifest.txt", help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, default="./outputs/hi", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    train(args.config, args.manifest, args.output_dir, args.epochs)
