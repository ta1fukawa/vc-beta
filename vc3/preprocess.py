import argparse
import pathlib
import sys

import torch
import torchaudio
import yaml

import common

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
            wave, sample_rate = common.norm_wave(wave, **config['norm_wave'])
            mel, _ = common.wave2mel(wave, sample_rate, **config['mel'])

            torch.save(mel, (mel_dir / speaker_name / f'{wav.stem}.pt'))

            mels.append(mel)

        embed = common.mel2embed(mels, encoder, device, **config['mel2embed'])

        torch.save(embed, (embed_dir / f'{speaker_name}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wav to mel spectrogram')
    parser.add_argument('wav_dir',      type=pathlib.Path, help='path to directory of speaker directories containing wav files')
    parser.add_argument('mel_dir',      type=pathlib.Path, help='path to directory to save mel spectrograms')
    parser.add_argument('embed_dir',    type=pathlib.Path, help='path to directory to save embeddings')
    parser.add_argument('encoder_path', type=pathlib.Path, help='path to speaker encoder')
    parser.add_argument('config_path',  type=pathlib.Path, help='path to config')

    if 'debugpy' in sys.modules:
        args = parser.parse_args([
            'autovc/wavs-jvs',
            'vc3/mel-jvs',
            'vc3/embed-jvs',
            'autovc2/dvector.pt',
            'vc3/preprocess.yaml',
        ])
    else:
        args = parser.parse_args([])

    main(**vars(args))
