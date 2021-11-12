import os

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def main():
    src_path = './resource/seiren_jvs011'
    tgt_path = './resource/seiren_jvs011_sp'
    tgt_path = './resource/sp/utterances'

    sr      = 24000
    nfft    = 1024
    hop_len = 256
    nmels   = 80

    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmels).T
    b, a = butter_highpass_filter(30, sr, order=5)  # バターワースフィルタ（ハイパスフィルタ）

    _, dir_list, _ = next(os.walk(src_path))

    for dir_name in sorted(dir_list):
        tgtdir_path = os.path.join(tgt_path, dir_name)
        os.makedirs(tgtdir_path, exist_ok=True)

        srcdir_path = os.path.join(src_path, dir_name)
        _, _, file_list = next(os.walk(srcdir_path))

        for file_name in sorted(file_list):
            file_path = os.path.join(srcdir_path, file_name)

            y, sr = sf.read(file_path)
            y = scipy.signal.filtfilt(b, a, y)  # ゼロ位相フィルタ（ドリフトノイズの除去）
            y = add_random_noise(y, 0.96, 1e-6)  # ノイズの追加
            sp = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop_len, window='hann')).T  # STFT
            mel = np.dot(sp, mel_basis)  # メルフィルタ
            db = 20 * np.log10(np.maximum(1e-5, mel)) - 16  # デシベル変換
            db = np.clip((db + 100) / 100, 0, 1)  # スケール調整（0～1に正規化）
            
            np.save(os.path.join(tgtdir_path, file_name.replace('.wav', '.npy')), db.astype(np.float32), allow_pickle=False)

def butter_highpass_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def add_random_noise(x, a, b):
    y = x * a + (np.random.rand(x.shape[0]) - 0.5) * b
    return y

if __name__ == '__main__':
    main()
