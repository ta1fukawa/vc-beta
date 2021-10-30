import numpy as np
import torch
import tqdm
import librosa

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, seiren_speaker_list, speaker_list, speech_list, batch_size=(4, 4), sp_length=1024, preload=False, seed=None, mel=False):
        self.seiren_speaker_list = seiren_speaker_list
        self.speaker_list        = speaker_list
        self.speech_list         = speech_list

        self.batch_size = batch_size
        self.sp_length  = sp_length
        self.preload    = preload
        self.mel_basis  = librosa.filters.mel(sr=24000, n_fft=1024) if mel else np.eye(512)

        if self.preload:
            assert sp_length is not None
            self.data = torch.empty((len(self.seiren_speaker_list), len(self.speaker_list), len(self.speech_list), sp_length, self.mel_basis.shape[0]), dtype=torch.float, device='cuda')
            bar = tqdm.tqdm(total=len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list))
            bar.set_description('Loading data')
            for seiren_speaker_idx, seiren_speaker in enumerate(self.seiren_speaker_list):
                for speaker_idx, speaker in enumerate(self.speaker_list):
                    for speech_idx, speech in enumerate(self.speech_list):
                        sp = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, (0 if mel else 1):]
                        sp = self._zero_padding(sp[:self.sp_length], self.sp_length)
                        sp = np.dot(sp, self.mel_basis.T)
                        self.data[seiren_speaker_idx][speaker_idx][speech_idx] = torch.from_numpy(np.array(sp)).float().to('cuda')
                        bar.update(1)
            bar.close()

        self.set_seed(seed)

    def __len__(self):
        return len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list) // np.prod(self.batch_size)

    def set_seed(self, seed=0):
        np.random.seed(seed if seed is not None else 0)
        if seed is not None:
            self.shuffle = np.random.permutation(len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list))
        else:
            self.shuffle = np.arange(len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list))

    def __getitem__(self, batch_idx):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()

        source_sp = torch.empty((self.batch_size[0], self.batch_size[1], self.sp_length, self.mel_basis.shape[0]), dtype=torch.float, device='cuda')
        batch_idx, speaker_offset = divmod(batch_idx, len(self.speaker_list) // self.batch_size[0])
        for speaker_idx in range(self.batch_size[0]):
            for i in range(self.batch_size[1]):
                temp = self.shuffle[batch_idx * self.batch_size[1] + i]
                temp, speech_idx         = divmod(temp, len(self.speech_list))
                temp, seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))

                source_sp_ij = self._load_sp(seiren_speaker_idx, speaker_offset * self.batch_size[0] + speaker_idx, speech_idx)
                source_sp[speaker_idx][i] = source_sp_ij
        # source_sp = torch.reshape(source_sp, (np.prod(self.batch_size), self.batch_size[1], self.sp_length, self.mel_basis.shape[0]))
        return source_sp

    def _load_sp(self, seiren_speaker_idx, speaker_idx, speech_idx):
        if self.preload:
            data = self.data[seiren_speaker_idx][speaker_idx][speech_idx]
        else:
            seiren_speaker = self.seiren_speaker_list[seiren_speaker_idx]
            speaker        = self.speaker_list[speaker_idx]
            speech         = self.speech_list[speech_idx]

            data = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, 1:]
            data = self._zero_padding(data[:self.sp_length], self.sp_length)
            data = torch.from_numpy(np.array(data)).float().to('cuda')

        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x
