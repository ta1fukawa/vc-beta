import csv
import os

import multiprocessing
import numpy as np
import pyworld
import soundfile as sf
import torch
from scipy import stats

class EmbedModel1d(torch.nn.Module):
    def __init__(self, n_freq, n_frames):
        super(EmbedModel1d, self).__init__()
        
        self.conv1a = torch.nn.Conv1d(n_freq, 512, kernel_size=3, dilation=1, padding='same')
        self.conv1b = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.drop1  = torch.nn.Dropout(p=0.2)
        
        self.conv2a = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.conv2b = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding='same')
        self.drop2  = torch.nn.Dropout(p=0.2)
        
        self.conv3  = torch.nn.Conv1d(512, 2048, kernel_size=3, dilation=1, padding='same')
        self.line3  = torch.nn.Linear(4096, 512)
        
    def _max_pooling(self, x):
        return x.max(dim=2)[0]
        
    def forward(self, x):
        x = torch.permute(x, dims=[0, 2, 1])
        
        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.drop1(x)
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self._max_pooling(x)
        x = self.line3(x)
        
        return x
        
class EmbedModel2d(torch.nn.Module):
    def __init__(self, n_freq, n_frames):
        super(EmbedModel2d, self).__init__()
        
        self.conv1a = torch.nn.Conv2d(1,  64, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop1  = torch.nn.Dropout2d(p=0.2)
        self.pool1  = torch.nn.MaxPool2d(kernel_size=(1, 4))
        
        self.conv2a = torch.nn.Conv2d(64,  128, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv2b = torch.nn.Conv2d(128, 128, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop2  = torch.nn.Dropout2d(p=0.2)
        self.pool2  = torch.nn.MaxPool2d(kernel_size=(1, 4))
        
        self.conv3a = torch.nn.Conv2d(128, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.conv3b = torch.nn.Conv2d(256, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.drop3  = torch.nn.Dropout2d(p=0.2)
        self.pool3  = torch.nn.MaxPool2d(kernel_size=(1, 4))
        
        self.conv4  = torch.nn.Conv2d(256, 2048, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.line4  = torch.nn.Linear(2048, 512)
        
    def _max_pooling(self, x):
        return x.max(dim=3)[0].max(dim=2)[0]
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        
        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.drop1(x)
        x = self.pool1(x)
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.drop2(x)
        x = self.pool2(x)
        
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.drop3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self._max_pooling(x)
        x = self.line4(x)
        
        return x
        
class FullModel(torch.nn.Module):
    def __init__(self, dim, n_freq=512, n_frames=32, nclasses=16):
        super(FullModel, self).__init__()
        
        if dim == 1:
            self.embed = EmbedModel1d(n_freq, n_frames)
        elif dim == 2:
            self.embed = EmbedModel2d(n_freq, n_frames)
        else:
            raise ValueError('引数dimは1～2である必要があります。')
        
        self.drop1 = torch.nn.Dropout(p=0.2)
        
        self.line2 = torch.nn.Linear(512, 1024)
        self.drop2 = torch.nn.Dropout(p=0.2)
        
        self.line3 = torch.nn.Linear(1024, nclasses)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.embed(x))
        x = self.drop1(x)
        
        x = torch.nn.functional.relu(self.line2(x))
        x = self.drop2(x)
        
        x = torch.nn.functional.log_softmax(self.line3(x), dim=-1)
        
        return x

def main():
    pool = multiprocessing.Pool(processes=16)
    speaker_embed_list = np.array(pool.map(get_speaker_embed, range(100)))
    np.save(f'resource/speaker-embeds.npz', embed=speaker_embed_list)
    pass

def get_speaker_embed(speaker):
    gpu_id = speaker % 4
    speaker_classfier = FullModel(2).to(f'cuda:{gpu_id}')
    speaker_classfier.load_state_dict(torch.load(f'resource/speaker-encoder.pth'))

    embed_list = list()
    for seiren_speaker in [10]:
        for speech in range(100):
            print(f'Processing: jvs{speaker + 1:03d} - seiren_jvs{seiren_speaker + 1:03d} - VOICEACTRESS100_{speech + 1:03d}\r', end='')

            wave, sr = sf.read(f'resource/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.wav')
            f0, sp, ap, t = wave_decompose(wave, sr)
            
            labels = load_csv(f'resource/jvs_ver1_fixed/jvs{seiren_speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.lab', delimiter='\t')
            phonemes = extract_phonemes(sp, labels)
            phonemes = torch.from_numpy(phonemes[:, :, :512]).float().to(f'cuda:{gpu_id}')

            embed_sub = speaker_classfier.embed(phonemes)
            embed_sub = embed_sub.to('cpu').detach().numpy().copy()
            embed_list.append(embed_sub)

    embed_list = np.vstack(embed_list)
    z = stats.gaussian_kde(embed_list.T)(embed_list.T)
    speaker_embed = embed_list[np.argmax(z)]
    return speaker_embed

def wave_decompose(wave, sr):
    f0, t = pyworld.harvest(wave, sr)
    sp = pyworld.cheaptrick(wave, f0, t, sr)
    ap = pyworld.d4c(wave, f0, t, sr)
    return f0, sp, ap, t

def load_csv(path, delimiter=','):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter=delimiter)
        labels = [row for row in tsv_reader]
    return labels

def extract_phonemes(sp, labels, target_length=32, min_length=24):
    sp_list = list()
    for start_sec, end_sec, phoneme in labels:
        if phoneme in ['silB', 'silE', 'sp']:
            continue

        separation_rate = 200
        start_frame = int(float(start_sec) * separation_rate)
        end_frame   = int(float(end_sec) * separation_rate)

        if start_frame + min_length + 1 >= end_frame:
            continue

        def padding(x, target_length):
            y_pad = target_length - len(x)
            return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x[:target_length]

        sp_list.append(padding(sp[start_frame:end_frame], target_length).astype(np.float32))
    return np.array(sp_list)

main()
