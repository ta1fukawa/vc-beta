import csv
import os
import pathlib
import sys

import librosa
import numpy as np
import scipy.signal
import soundfile as sf

sys.path.append('./julius4seg')

from julius4seg.sp_inserter import ModelType
from sample.run_segment import run_segment


def main():
    src_path = './resource/seiren_jvs011'
    tgt_path = './resource/wav/seiren_jvs011_slow'

    sr      = 24000
    nfft    = 1024
    hop_len = 256
    
    _, dir_list, _ = next(os.walk(src_path))

    for dir_name in sorted(dir_list):
        tgtdir_path = os.path.join(tgt_path, dir_name)
        os.makedirs(tgtdir_path, exist_ok=True)

        srcdir_path = os.path.join(src_path, dir_name)
        _, _, file_list = next(os.walk(srcdir_path))

        for file_name in sorted(file_list):
            file_path = os.path.join(srcdir_path, file_name)

            y, sr = sf.read(file_path)
            
            sp = librosa.stft(y, n_fft=nfft, hop_length=hop_len, window='hann')
            sp = librosa.phase_vocoder(sp, 0.5, hop_length=hop_len)
            y  = librosa.istft(sp, hop_length=hop_len, window='hann')

            sf.write(os.path.join(tgtdir_path, file_name), y, sr)

if __name__ == '__main__':
    main()
