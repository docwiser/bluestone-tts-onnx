"""
PyTorch Dataset and Audio Processing for Student TTS Training.
Computes Linear Spectrogram and Mel-Spectrograms on-the-fly.
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from frontend.indic_tokenizer import IndicTokenizer

class MelSpectrogramComputer:
    def __init__(self, sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80, f_min=0.0, f_max=8000.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=1.0,
            normalized=False
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1, T]
        mel = self.mel_transform(wav)
        # Log mel compression with dynamic range clipping
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0)  # [n_mels, T_mel]

    def spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        # Linear spectrogram for posterior encoder
        window = torch.hann_window(self.win_length, device=wav.device)
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )
        spec = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)
        return spec.squeeze(0)  # [n_fft // 2 + 1, T_spec]

class DistilledTTSDataset(Dataset):
    """
    Dataset that loads paired text and audio (either teacher generated or ground-truth),
    with optional precomputed teacher durations.
    """
    def __init__(self, manifest_file: str, tokenizer: IndicTokenizer, audio_cfg: dict):
        self.items = []
        self.tokenizer = tokenizer
        self.sample_rate = audio_cfg["sampling_rate"]
        self.mel_computer = MelSpectrogramComputer(
            sample_rate=self.sample_rate,
            n_fft=audio_cfg["filter_length"],
            win_length=audio_cfg["win_length"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mel_channels"],
            f_min=audio_cfg["mel_fmin"],
            f_max=audio_cfg["mel_fmax"]
        )

        base_dir = os.path.dirname(os.path.abspath(manifest_file))
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    wav_path = parts[0]
                    text = parts[1]
                    dur_path = parts[2] if len(parts) >= 3 else None
                    if not os.path.isabs(wav_path):
                        wav_path = os.path.join(base_dir, wav_path)
                    if os.path.exists(wav_path):
                        self.items.append((wav_path, text, dur_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text, dur_path = self.items[idx]

        # 1. Text tokenization
        text_ids = self.tokenizer.encode(text, add_bos_eos=True, add_blank=True)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # 2. Audio loading & resampling if necessary
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)

        # Normalize waveform
        wav = wav / (torch.max(torch.abs(wav)) + 1e-6) * 0.95

        # 3. Spectrogram & Mel computation
        spec = self.mel_computer.spectrogram(wav)
        mel = self.mel_computer(wav)

        # 4. Optional duration
        durations = None
        if dur_path and os.path.exists(dur_path):
            durs = np.load(dur_path)
            durations = torch.from_numpy(durs).float()

        return text_tensor, spec, mel, wav.squeeze(0), durations

def collate_fn(batch):
    """Collates and pads variable length batches."""
    batch_size = len(batch)

    text_lens = torch.tensor([item[0].size(0) for item in batch], dtype=torch.long)
    spec_lens = torch.tensor([item[1].size(1) for item in batch], dtype=torch.long)
    wav_lens = torch.tensor([item[3].size(0) for item in batch], dtype=torch.long)

    max_text_len = text_lens.max().item()
    max_spec_len = spec_lens.max().item()
    max_wav_len = wav_lens.max().item()

    spec_dim = batch[0][1].size(0)
    mel_dim = batch[0][2].size(0)

    padded_text = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    padded_spec = torch.zeros(batch_size, spec_dim, max_spec_len, dtype=torch.float32)
    padded_mel = torch.zeros(batch_size, mel_dim, max_spec_len, dtype=torch.float32)
    padded_wav = torch.zeros(batch_size, 1, max_wav_len, dtype=torch.float32)

    has_durations = batch[0][4] is not None
    padded_durs = torch.zeros(batch_size, max_text_len, dtype=torch.float32) if has_durations else None

    for i, (text, spec, mel, wav, durs) in enumerate(batch):
        padded_text[i, :text.size(0)] = text
        padded_spec[i, :, :spec.size(1)] = spec
        padded_mel[i, :, :mel.size(1)] = mel
        padded_wav[i, 0, :wav.size(0)] = wav
        if has_durations and durs is not None:
            padded_durs[i, :min(durs.size(0), max_text_len)] = durs[:max_text_len]

    return {
        "text": padded_text,
        "text_lengths": text_lens,
        "spec": padded_spec,
        "spec_lengths": spec_lens,
        "mel": padded_mel,
        "wav": padded_wav,
        "wav_lengths": wav_lens,
        "durations": padded_durs
    }
