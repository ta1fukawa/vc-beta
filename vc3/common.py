import torch
import torchaudio

def norm_wave(wave, sample_rate, norm_db, sil_threshold, sil_duration, preemph):
    effects = [
        ["channels", "1"],
        ["rate", f"{sample_rate}"],
        ["norm", f"{norm_db}"],
        [
            "silence",
            "1",
            f"{sil_duration}",
            f"{sil_threshold}%",
            "-1",
            f"{sil_duration}",
            f"{sil_threshold}%",
        ],
    ]

    wave, sample_rate = torchaudio.sox_effects.apply_effects_tensor(wave, sample_rate, effects)
    wave = torch.cat([wave[:, 0].unsqueeze(-1), wave[:, 1:] - preemph * wave[:, :-1]], dim=-1)

    return wave, sample_rate

def wave2mel(wave, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms    / 1000),
        power=None,
    )(wave)
    spec  = torch.abs(spec)**2
    angle = torch.angle(spec)
    
    mel = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        n_stft=n_fft // 2 + 1,
    )(spec).T

    mel = 20 * torch.log10(torch.clamp(mel, min=1e-9))
    mel = (mel - ref_db) / (dc_db - ref_db)

    return mel, angle

def mel2wave(mel, angle, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
    mel = mel * (dc_db - ref_db) + ref_db
    mel = 10 ** (mel / 20)
    spec = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
    )(mel.T)
    spec = spec * torch.cos(angle) + spec * torch.sin(angle)
    wave = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms    / 1000),
    )(spec.unsqueeze(0))

    return wave

def mel2embed(mels, encoder, device, seg_len):
    mels = torch.stack([mel[:seg_len] for mel in mels if len(mel) >= seg_len], dim=0).to(device)

    with torch.no_grad():
        embeds = encoder(mels)

    embed = torch.mean(embeds, dim=0)
    return embed

def pad_seq(mel, seg_len):
    if len(mel) < seg_len:
        len_pad = seg_len - len(mel)
        mel = torch.cat((mel, torch.zeros(len_pad, mel.shape[1])), dim=0)
    else:
        mel = mel[:seg_len]
    return mel