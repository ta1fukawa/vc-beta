import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from synthesis import build_model, wavegen

from model import AutoVC


def main():
    device = 'cuda:0'

    uttr_path = './resource/seiren_jvs011_sp/jvs001/VOICEACTRESS100_001.npy'
    uttr = np.load(uttr_path)
    pad = 1024 - uttr.shape[0]
    if pad > 0:
        uttr = np.concatenate([uttr, np.zeros((pad, uttr.shape[1]))])
    else:
        uttr = uttr[:1024]
    src_sp = torch.from_numpy(uttr[None]).float().to(device)

    embs = np.load('./resource/emb/emb3_centroids.npy', allow_pickle=True)
    embs = torch.from_numpy(embs[None]).float().to(device)
    src_emb = embs[:, 0]
    tgt_emb = embs[:, 0]

    model = AutoVC(16, 16).to(device).eval()
    model.load_state_dict(torch.load('./dest/test-04/weights.pth'))
    with torch.no_grad():
        _, tgt_sp, _ = model(src_sp, src_emb, tgt_emb)
        tgt_sp = tgt_sp.cpu().numpy()[0][0]
    
    print(np.mean(np.abs(uttr - tgt_sp)))
    pass

    # sr      = 24000
    # nfft    = 1024
    # hop_len = 256
    # nmels   = 80

    # mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmels).T
    # mel_basis_inv = np.linalg.pinv(mel_basis)

    # org_wave, sr = sf.read('./resource/seiren_jvs011/jvs001/VOICEACTRESS100_001.wav')
    # phase = np.angle(librosa.stft(org_wave, n_fft=nfft, hop_length=hop_len, window='hann'))[:, :1024]

    # db  = tgt_sp[:phase.shape[1]] * 100 - 100
    # mel = 10**((db + 16) / 20)
    # sp  = np.dot(mel, mel_basis_inv)

    # wave = librosa.istft(sp.T * np.exp(1j * phase), hop_length=hop_len, window='hann')

    # # vocoder = build_model().to(device).eval()
    # # vocoder.load_state_dict(torch.load("./autovc/checkpoint_step001000000_ema.pth")["state_dict"])

    # # wave = wavegen(vocoder, c=tgt_sp)
    # sf.write('./dest/test-04/001.wav', wave, 24000)

if __name__ == '__main__':
    main()
