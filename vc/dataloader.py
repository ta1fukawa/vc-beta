import numpy as np
import torch

class DataLoader4AE(torch.utils.data.Dataset):
    def __init__(self, seiren_speaker_list, speaker_list, speech_list, batch_size=1, sp_length=1024, preload=False, seed=None):
        self.seiren_speaker_list = seiren_speaker_list
        self.speaker_list        = speaker_list
        self.speech_list         = speech_list
        self.speaker_embeds = np.load('resource/speaker-embeds.npz', allow_pickle=True)['embed']
        self.batch_size = batch_size
        self.sp_length  = sp_length
        self.preload    = preload

        if self.preload:
            assert sp_length is not None
            self.data = np.empty((len(self.seiren_speaker_list), len(self.speaker_list), len(self.speech_list), sp_length, 512))
            for seiren_speaker_idx, seiren_speaker in enumerate(self.seiren_speaker_list):
                for speaker_idx, speaker in enumerate(self.speaker_list):
                    for speech_idx, speech in enumerate(self.speech_list):
                        sp = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, :512]
                        sp = self._zero_padding(sp[:self.sp_length], self.sp_length)
                        self.data[seiren_speaker_idx][speaker_idx][speech_idx] = sp

        self.set_seed(seed)

    def __len__(self):
        return len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list)

    def set_seed(self, seed=0):
        np.random.seed(seed if seed is not None else 0)
        if seed is not None:
            self.shuffle = np.random.permutation(len(self))
        else:
            self.shuffle = np.arange(len(self))

    def __getitem__(self, batch_idx):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()
        
        source_sp = list()
        speaker_embed = list()
        for i in range(self.batch_size):
            temp = self.shuffle[batch_idx * self.batch_size + i]
            temp, source_speech_idx         = divmod(temp, len(self.speech_list))
            temp, source_speaker_idx        = divmod(temp, len(self.speaker_list))
            temp, source_seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))

            source_sp_i = self._load_sp(source_seiren_speaker_idx, source_speaker_idx, source_speech_idx)
            speaker_embed_i = self.speaker_embeds[source_speaker_idx]

            if not self.preload and self.sp_length is not None:
                source_sp_i = self._zero_padding(source_sp_i[:self.sp_length], self.sp_length)

            source_sp.append(source_sp_i)
            speaker_embed.append(speaker_embed_i)

        source_sp = torch.from_numpy(np.array(source_sp)).float().to('cuda')
        speaker_embed = torch.from_numpy(np.array(speaker_embed)).float().to('cuda')
        return source_sp, source_sp, speaker_embed
    
    def _load_sp(self, seiren_speaker_idx, speaker_idx, speech_idx):
        seiren_speaker = self.seiren_speaker_list[seiren_speaker_idx]
        speaker        = self.speaker_list[speaker_idx]
        speech         = self.speech_list[speech_idx]
        if self.preload:
            data = self.data[seiren_speaker_idx][speaker_idx][speech_idx]
        else:
            data = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, :512]
        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x

class DataLoader4FL(torch.utils.data.Dataset):
    def __init__(self, seiren_speaker_list, speaker_list, speech_list, batch_size=1, sp_length=1024, preload=False, seed=None):
        self.seiren_speaker_list = seiren_speaker_list
        self.speaker_list        = speaker_list
        self.speech_list         = speech_list
        self.speaker_embeds = np.load('resource/emb/xvec_centroids.npz', allow_pickle=True)['centroids']
        self.batch_size = batch_size
        self.sp_length  = sp_length
        self.preload    = preload

        if self.preload:
            assert sp_length is not None
            self.data = np.empty((len(self.seiren_speaker_list), len(self.speaker_list), len(self.speech_list), sp_length, 512))
            for seiren_speaker_idx, seiren_speaker in enumerate(self.seiren_speaker_list):
                for speaker_idx, speaker in enumerate(self.speaker_list):
                    for speech_idx, speech in enumerate(self.speech_list):
                        sp = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, :512]
                        sp = self._zero_padding(sp[:self.sp_length], self.sp_length)
                        self.data[seiren_speaker_idx][speaker_idx][speech_idx] = sp

        self.set_seed(seed)

    def __len__(self):
        # return (len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list))**2
        return len(self.seiren_speaker_list) * len(self.speaker_list) * len(self.speech_list)

    def set_seed(self, seed=0):
        np.random.seed(seed if seed is not None else 0)
        if seed is not None:
            self.shuffle = np.random.permutation(len(self))
        else:
            self.shuffle = np.arange(len(self))

    def __getitem__(self, batch_idx):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()
        
        source_sp = list()
        target_sp = list()
        speaker_embed = list()
        for i in range(self.batch_size):
            temp = self.shuffle[batch_idx * self.batch_size + i]
            temp, target_speech_idx         = divmod(temp, len(self.speech_list))
            temp, target_speaker_idx        = divmod(temp, len(self.speaker_list))
            temp, target_seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))
            temp, source_speech_idx         = divmod(temp, len(self.speech_list))
            temp, source_speaker_idx        = divmod(temp, len(self.speaker_list))
            temp, source_seiren_speaker_idx = divmod(temp, len(self.seiren_speaker_list))

            source_sp_i = self._load_sp(source_seiren_speaker_idx, source_speaker_idx, source_speech_idx)
            target_sp_i = self._load_sp(target_seiren_speaker_idx, target_speaker_idx, target_speech_idx)
            speaker_embed_i = self.speaker_embeds[target_speaker_idx]

            if not self.preload and self.sp_length is not None:
                source_sp_i = self._zero_padding(source_sp_i[:self.sp_length], self.sp_length)
                target_sp_i = self._zero_padding(target_sp_i[:self.sp_length], self.sp_length)

            source_sp.append(source_sp_i)
            target_sp.append(target_sp_i)
            speaker_embed.append(speaker_embed_i)

        source_sp = torch.from_numpy(np.array(source_sp)).float().to('cuda')
        target_sp = torch.from_numpy(np.array(target_sp)).float().to('cuda')
        speaker_embed = torch.from_numpy(np.array(speaker_embed)).float().to('cuda')
        return source_sp, target_sp, speaker_embed
    
    def _load_sp(self, seiren_speaker_idx, speaker_idx, speech_idx):
        seiren_speaker = self.seiren_speaker_list[seiren_speaker_idx]
        speaker        = self.speaker_list[speaker_idx]
        speech         = self.speech_list[speech_idx]
        if self.preload:
            data = self.data[seiren_speaker_idx][speaker_idx][speech_idx]
        else:
            data = np.load(f'resource/mid/seiren_jvs{seiren_speaker + 1:03d}/jvs{speaker + 1:03d}/VOICEACTRESS100_{speech + 1:03d}.npz', allow_pickle=True)['sp'][:, :512]
        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x
