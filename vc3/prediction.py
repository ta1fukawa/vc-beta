import argparse
import pathlib
import sys

import torch
import torchaudio
import yaml

import common

def main(source, target, output, encoder_path, model_dir, vocoder_path, config_path):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = torch.load(encoder_path)          .to(device).eval()
    model   = torch.load(model_dir / 'model.pt').to(device).eval()
    # vocoder = torch.load(vocoder_path)          .to(device).eval()
    config  = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    model.load_state_dict(torch.load(model_dir / 'weight.pt'))

    output.parent.mkdir(exist_ok=True, parents=True)

    src_wave, sample_rate = torchaudio.load(source)
    tgt_wave, sample_rate = torchaudio.load(target)
    src_wave, sample_rate = common.norm_wave(src_wave, **config['norm_wave'])
    tgt_wave, sample_rate = common.norm_wave(tgt_wave, **config['norm_wave'])
    src_mel, src_angle = common.wave2mel(src_wave, sample_rate, **config['mel'])
    tgt_mel, tgt_angle = common.wave2mel(tgt_wave, sample_rate, **config['mel'])
    src_emb = common.mel2embed(src_mel.unsqueeze(0).to(device), encoder, device, **config['mel2embed']).unsqueeze(0)
    tgt_emb = common.mel2embed(tgt_mel.unsqueeze(0).to(device), encoder, device, **config['mel2embed']).unsqueeze(0)

    src_mel = common.pad_seq(src_mel, config['mel2embed']['seg_len']).unsqueeze(0)
    with torch.no_grad():
        _, _, pred_mel, _ = model(src_mel, src_emb, tgt_emb)
        # pred_wave         = vocoder.generate([pred_mel.squeeze(0)])[0].data.unsqueeze(0).cpu()

    pred_wave = common.mel2wave(pred_mel.squeeze(0).cpu(), src_angle, sample_rate, **config['mel'])
    torchaudio.save(output, pred_wave, sample_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wav to mel spectrogram')
    parser.add_argument('source',       type=pathlib.Path, help='path to source wav')
    parser.add_argument('target',       type=pathlib.Path, help='path to target wav')
    parser.add_argument('output',       type=pathlib.Path, help='path to output wav')
    parser.add_argument('encoder_path', type=pathlib.Path, help='path to speaker encoder')
    parser.add_argument('model_dir',    type=pathlib.Path, help='path to model directory')
    parser.add_argument('vocoder_path', type=pathlib.Path, help='path to vocoder path')
    parser.add_argument('config_path',  type=pathlib.Path, help='path to config')

    if 'debugpy' in sys.modules:
        args = parser.parse_args([
            'autovc/wavs-jvs/jvs001/VOICEACTRESS100_001.wav',
            'autovc/wavs-jvs/jvs002/VOICEACTRESS100_002.wav',
            'vc3/pred/output.wav',
            'autovc2/dvector.pt',
            'vc3/train/2021-12-18/21-44-11',
            'autovc2/vocoder.pt',
            'vc3/preprocess.yaml',
        ])
    else:
        args = parser.parse_args([])

    main(**vars(args))
