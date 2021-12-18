import argparse
import pathlib
import sys

import torch
import torchaudio
import yaml


def wave2mel(wave, sample_rate, norm_db, sil_threshold, sil_duration, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, preemph, ref_db, dc_db):
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

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms    / 1000),
        n_fft=n_fft,
        f_min=f_min,
        n_mels=n_mels,
    )(wave).squeeze(0).T  # (time, n_mels)
    mel = 20 * torch.log10(torch.clamp(mel, min=1e-9))
    mel = (mel - ref_db) / (dc_db - ref_db)

    return mel


def mel2embed(mels, encoder, device, seg_len):
    mels = torch.stack([mel[:seg_len] for mel in mels if len(mel) >= seg_len], dim=0).to(device)

    with torch.no_grad():
        embeds = encoder(mels)

    embed = torch.mean(embeds, dim=0)
    return embed

def main(wav_dir, mel_dir, embed_dir, encoder_path, config_path):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = torch.load(encoder_path).to(device).eval()
    config  = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)

    embed_dir.mkdir(exist_ok=True, parents=True)

    for speaker in wav_dir.iterdir():
        if not speaker.is_dir():
            continue

        speaker_name = speaker.name

        (mel_dir / speaker_name).mkdir(exist_ok=True, parents=True)

        mels = []

        for wav in speaker.iterdir():
            if not wav.is_file() or wav.suffix != '.wav':
                continue

            wave, sample_rate = torchaudio.load(wav)
            mel   = wave2mel(wave, **config['wave2mel'])

            torch.save(mel, (mel_dir / speaker_name / f'{wav.stem}.pt'))

            mels.append(mel)

        embed = mel2embed(mels, encoder, device, **config['mel2embed'])

        torch.save(embed, (embed_dir / f'{speaker_name}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wav to mel spectrogram')
    parser.add_argument('wav_dir',      type=pathlib.Path, help='directory of speaker directories containing wav files')
    parser.add_argument('mel_dir',      type=pathlib.Path, help='directory to save mel spectrograms')
    parser.add_argument('embed_dir',    type=pathlib.Path, help='directory to save embeddings')
    parser.add_argument('encoder_path', type=pathlib.Path, help='path to speaker encoder')
    parser.add_argument('config_path',  type=pathlib.Path, help='path to config')

    if 'debugpy' in sys.modules:
        args = parser.parse_args([
            'autovc/wavs-jvs',
            'vc3/mel-jvs',
            'vc3/embed-jvs',
            'autovc2/dvector.pt',
            'vc3/preprocess.yaml'
        ])
    else:
        args = parser.parse_args()

    main(**vars(args))
