import csv
import os
import pathlib
import sys

import librosa
import numpy as np
import pyworld
import scipy.signal
import soundfile as sf

sys.path.append('./julius4seg')

from julius4seg.sp_inserter import ModelType
from sample.run_segment import run_segment


def main():
    src_path = './resource/wav/seiren_jvs011_slow'
    tgt_path = './resource/spenv/phonemes_v1'
    
    julius_src_path = './resource/wav/seiren_jvs011_slow/jvs011'
    lab_yomi_path   = './resource/voiceactoress100_spaced_julius.txt'
    lab_path        = './resource/lab/seiren_jvs011_slow'
    hmm_path        = 'resource/dictation-kit-4.5/model/phone_m/jnas-mono-16mix-gid.binhmm'

    use_env = True
    use_mel = False

    sr      = 24000
    nfft    = 1024
    hop_len = 256
    nmels   = 80
    
    if not os.path.exists(lab_path):
        os.makedirs(lab_path)

        with open(lab_yomi_path, 'r') as f:
            yomi_list = f.readlines()

        _, _, file_list = next(os.walk(julius_src_path))

        for idx, file_name in enumerate(sorted(file_list)):
            file_path = os.path.join(julius_src_path, file_name)

            with open(os.path.join('/tmp', file_name.replace('.wav', '.txt')), 'w') as f:
                f.write(yomi_list[idx])

            wave, sr = librosa.load(file_path, sr=16000, mono=True)
            sf.write(os.path.join('/tmp', file_name), wave, sr, subtype='PCM_16')

            julius4seg_args = {
                'wav_file': pathlib.Path(os.path.join('/tmp', file_name)),
                'input_yomi_file': pathlib.Path(os.path.join('/tmp', file_name.replace('.wav', '.txt'))),
                'output_seg_file': pathlib.Path(os.path.join(lab_path, file_name.replace('.wav', '.lab'))),
                'input_yomi_type': 'katakana',
                'like_openjtalk': False,
                'input_text_file': None,
                'output_text_file': None,
                'hmm_model': hmm_path,
                'model_type': ModelType.gmm,
                'padding_second': 0,
                'options': None
            }

            try:
                run_segment(**julius4seg_args, only_2nd_path=False)
            except:
                run_segment(**julius4seg_args, only_2nd_path=True)
    
    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmels).T
    b, a = butter_highpass_filter(30, sr, order=5)  # バターワースフィルタ（ハイパスフィルタ）

    _, dir_list, _ = next(os.walk(src_path))

    file_lens = {}

    for dir_name in sorted(dir_list):
        tgtdir_path = os.path.join(tgt_path, dir_name)
        os.makedirs(tgtdir_path, exist_ok=True)

        srcdir_path = os.path.join(src_path, dir_name)
        _, _, file_list = next(os.walk(srcdir_path))

        phnm_idx = 1
        for file_name in sorted(file_list):
            file_path = os.path.join(srcdir_path, file_name)

            y, sr = sf.read(file_path)
            y = scipy.signal.filtfilt(b, a, y)   # ゼロ位相フィルタ（ドリフトノイズの除去）
            y = add_random_noise(y, 0.96, 1e-6)  # ノイズの追加
            
            if use_env:
                f0, t  = pyworld.harvest(y, sr)        # 基本周波数の抽出
                sp = pyworld.cheaptrick(y, f0, t, sr)  # スペクトル包絡の抽出 spectrogram (n, f)
            else:
                sp = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop_len, window='hann')).T  # STFT

            if use_mel:
                sp = np.dot(sp, mel_basis)                 # メルフィルタ
            sp = 20 * np.log10(np.maximum(1e-5, sp)) - 16  # デシベル変換
            sp = np.clip((sp + 100) / 100, 0, 1)           # スケール調整（0～1に正規化）

            if file_name not in file_lens:
                file_lens[file_name] = len(sp)

            with open(os.path.join(lab_path, file_name.replace('.wav', '.lab')), 'r', newline='', encoding='utf-8') as f:
                tsv_reader = csv.reader(f, delimiter='\t')
                labels = [row for row in tsv_reader]

            for start, end, label in labels:
                if label in ['silB', 'silE', 'sp']:
                    continue

                start = int(float(start) * file_lens[file_name] / float(labels[-1][1]))
                end   = int(float(end)   * file_lens[file_name] / float(labels[-1][1]) + 1)
                if start + 32 + 1 >= end:
                    continue
                
                sp_seg = sp[start:end]
                tgt_file_path = os.path.join(tgtdir_path, f'phoneme_{phnm_idx:04d}.npy')
                np.save(tgt_file_path, sp_seg.astype(np.float32), allow_pickle=False)

                phnm_idx += 1

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
