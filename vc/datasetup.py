import os
import multiprocessing

import numpy as np
import pyworld
import soundfile as sf

def first_setup():
    pool_obj = multiprocessing.Pool()
    pool_obj.map(first_setup_sub, range(100))

def first_setup_sub(speaker):
    for seiren_speaker in [10]:
        os.makedirs(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}', exist_ok=True)
        for speech in range(100):
            wave, sr = sf.read(f'resource/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.wav')
            f0, sp, ap, t = wave_decompose(wave, sr)
            np.savez_compressed(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', sp=sp)

def wave_decompose(wave, sr):
    f0, t = pyworld.harvest(wave, sr)
    sp = pyworld.cheaptrick(wave, f0, t, sr)
    ap = pyworld.d4c(wave, f0, t, sr)
    return f0, sp, ap, t