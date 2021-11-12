import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch

from vc_model import AutoVCConv2d


# テスト用の仮コード

def main():
    device = 'cuda:1'

    src_sp = load_sp('./resource/seiren_jvs011_sp/jvs001/VOICEACTRESS100_001.npy')
    src_sp = torch.from_numpy(src_sp[None]).float().to(device)

    # true_sp = load_sp('./resource/seiren_jvs011_sp/jvs001/VOICEACTRESS100_002.npy')

    embs = np.load('./dest/emb-main/20211104-000810/centroids.npy', allow_pickle=True)
    embs = torch.from_numpy(embs[None]).float().to(device)
    src_emb = embs[:, 0]
    tgt_emb = embs[:, 0]

    model = AutoVCConv2d(512, 512, 80).to(device).eval()
    model.load_state_dict(torch.load('./dest/vc-train/20211104-170318/weights.pth'))
    with torch.no_grad():
        _, tgt_sp, _ = model(src_sp, src_emb, tgt_emb)
        tgt_sp = tgt_sp.cpu().numpy()[0]
    print(tgt_sp.shape)
    
    # print(np.mean(np.abs(true_sp - tgt_sp)))
    # print(np.mean(np.abs(true_sp)))
    # print(np.mean(np.abs(tgt_sp)))

    sr      = 24000
    nfft    = 1024
    hop_len = 256
    nmels   = 80

    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmels).T
    mel_basis_inv = np.linalg.pinv(mel_basis)

    org_wave, sr = sf.read('./resource/seiren_jvs011/jvs001/VOICEACTRESS100_001.wav')
    phase = np.angle(librosa.stft(org_wave, n_fft=nfft, hop_length=hop_len, window='hann'))[:, :512]

    db  = tgt_sp[:phase.shape[1]] * 100 - 100
    mel = 10**((db + 16) / 20)
    sp  = np.dot(mel, mel_basis_inv)

    wave = librosa.istft(sp.T * np.exp(1j * phase), hop_length=hop_len, window='hann')

    # vocoder = build_model().to(device).eval()
    # vocoder.load_state_dict(torch.load("./autovc/checkpoint_step001000000_ema.pth")["state_dict"])

    # wave = wavegen(vocoder, c=tgt_sp)
    sf.write('./dest/test/test-04/007.wav', wave, 24000)

def load_sp(path):
    sp = np.load(path)
    pad = 512 - sp.shape[0]
    if pad > 0:
        sp = np.concatenate([sp, np.zeros((pad, sp.shape[1]))])
    else:
        sp = sp[:512]
    return sp

if __name__ == '__main__':
    main()
