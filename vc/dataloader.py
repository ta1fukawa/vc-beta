import numpy as np
import torch

class DataLoaderSame(torch.utils.data.Dataset):
    def __init__(self, seiren_speaker_list, speaker_list, speech_list, batch_size=1, sp_length=1024, preload=False, seed=None):
        self.seiren_speaker_list = seiren_speaker_list
        self.speaker_list        = speaker_list
        self.speech_list         = speech_list

        self.speaker_embeds = np.load('resource/emb/xvec_centroids.npz', allow_pickle=True)['centroids']
        self.speaker_embeds = self.speaker_embeds[self.speaker_list]
        self.speaker_embeds = torch.from_numpy(self.speaker_embeds).float().to('cuda')

        self.batch_size = batch_size
        self.sp_length  = sp_length
        self.preload    = preload

        if self.preload:
            assert sp_length is not None
            self.data = np.empty((len(self.seiren_speaker_list), len(self.speaker_list), len(self.speech_list), sp_length, 512))
            for seiren_speaker_idx, seiren_speaker in enumerate(self.seiren_speaker_list):
                for speaker_idx, speaker in enumerate(self.speaker_list):
                    for speech_idx, speech in enumerate(self.speech_list):
                        sp = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, 1:]
                        sp = self._zero_padding(sp[:self.sp_length], self.sp_length)
                        self.data[seiren_speaker_idx][speaker_idx][speech_idx] = sp
            self.data = torch.from_numpy(np.array(self.data)).float().to('cuda')

        self.set_seed(seed)

    def __len__(self):
        return len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list) // self.batch_size

    def set_seed(self, seed=0):
        np.random.seed(seed if seed is not None else 0)
        if seed is not None:
            self.shuffle = np.random.permutation(len(self) * self.batch_size)
        else:
            self.shuffle = np.arange(len(self) * self.batch_size)

    def __getitem__(self, batch_idx):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()
        
        source_sp = list()
        speaker_embed = list()
        for i in range(self.batch_size):
            temp = self.shuffle[batch_idx * self.batch_size + i]
            temp, source_speaker_idx = divmod(temp, len(self.speaker_list))
            temp, speech_idx         = divmod(temp, len(self.speech_list))
            temp, seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))

            source_sp_i = self._load_sp(seiren_speaker_idx, source_speaker_idx, speech_idx)
            speaker_embed_i = self.speaker_embeds[source_speaker_idx]

            source_sp.append(source_sp_i)
            speaker_embed.append(speaker_embed_i)

        source_sp = torch.stack(source_sp)
        speaker_embed = torch.stack(speaker_embed)
        return source_sp, source_sp, speaker_embed
    
    def _load_sp(self, seiren_speaker_idx, speaker_idx, speech_idx):
        if self.preload:
            data = self.data[seiren_speaker_idx][speaker_idx][speech_idx]
        else:
            seiren_speaker = self.seiren_speaker_list[seiren_speaker_idx]
            speaker        = self.speaker_list[speaker_idx]
            speech         = self.speech_list[speech_idx]

            data = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, 1:]
            if self.sp_length is not None:
                data = self._zero_padding(data[:self.sp_length], self.sp_length)

        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x

class DataLoaderVary(torch.utils.data.Dataset):
    def __init__(self, seiren_speaker_list, speaker_list, speech_list, batch_size=1, sp_length=1024, preload=False, seed=None):
        self.seiren_speaker_list = seiren_speaker_list
        self.speaker_list        = speaker_list
        self.speech_list         = speech_list

        self.speaker_embeds = np.load('resource/emb/xvec_centroids.npz', allow_pickle=True)['centroids']
        self.speaker_embeds = self.speaker_embeds[self.speaker_list]
        self.speaker_embeds = torch.from_numpy(self.speaker_embeds).float().to('cuda')

        self.batch_size = batch_size
        self.sp_length  = sp_length
        self.preload    = preload

        if self.preload:
            assert sp_length is not None
            self.data = torch.empty((len(self.seiren_speaker_list), len(self.speaker_list), len(self.speech_list), sp_length, 512), dtype=torch.float, device='cuda')
            for seiren_speaker_idx, seiren_speaker in enumerate(self.seiren_speaker_list):
                for speaker_idx, speaker in enumerate(self.speaker_list):
                    for speech_idx, speech in enumerate(self.speech_list):
                        sp = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, 1:]
                        sp = self._zero_padding(sp[:self.sp_length], self.sp_length)
                        self.data[seiren_speaker_idx][speaker_idx][speech_idx] = torch.from_numpy(np.array(sp)).float().to('cuda')

        self.set_seed(seed)

    def __len__(self):
        return len(self.seiren_speaker_list) * len(self.speaker_list)**2 * len(self.speech_list) // self.batch_size

    def set_seed(self, seed=0):
        np.random.seed(seed if seed is not None else 0)
        if seed is not None:
            self.shuffle = np.random.permutation(len(self) * self.batch_size)
        else:
            self.shuffle = np.arange(len(self) * self.batch_size)

    def __getitem__(self, batch_idx):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()
        
        source_sp = list()
        target_sp = list()
        speaker_embed = list()
        for i in range(self.batch_size):
            temp = self.shuffle[batch_idx * self.batch_size + i]
            temp, source_speaker_idx = divmod(temp, len(self.speaker_list))
            temp, target_speaker_idx = divmod(temp, len(self.speaker_list))
            temp, speech_idx         = divmod(temp, len(self.speech_list))
            temp, seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))

            source_sp_i = self._load_sp(seiren_speaker_idx, source_speaker_idx, speech_idx)
            target_sp_i = self._load_sp(seiren_speaker_idx, target_speaker_idx, speech_idx)
            speaker_embed_i = self.speaker_embeds[target_speaker_idx]

            source_sp.append(source_sp_i)
            target_sp.append(target_sp_i)
            speaker_embed.append(speaker_embed_i)

        source_sp = torch.stack(source_sp)
        target_sp = torch.stack(target_sp)
        speaker_embed = torch.stack(speaker_embed)
        return source_sp, target_sp, speaker_embed
    
    def _load_sp(self, seiren_speaker_idx, speaker_idx, speech_idx):
        if self.preload:
            data = self.data[seiren_speaker_idx][speaker_idx][speech_idx]
        else:
            seiren_speaker = self.seiren_speaker_list[seiren_speaker_idx]
            speaker        = self.speaker_list[speaker_idx]
            speech         = self.speech_list[speech_idx]

            data = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, 1:]
            data = self._zero_padding(data[:self.sp_length], self.sp_length)

        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x
