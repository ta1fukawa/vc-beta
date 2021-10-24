import numpy as np
import torch
import librosa

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, person_list, voice_list, batch_size, dataset_path, deform_type, phonemes_length=None, mel=False, seed=None):
        self.person_list = person_list
        self.voice_list  = voice_list
        self.batch_size  = batch_size

        self.dataset_path    = dataset_path
        self.deform_type     = deform_type
        self.phonemes_length = phonemes_length
        self.mel_basis       = librosa.filters.mel(sr=24000, n_fft=1024) if mel else None

        self.person_nbatches  = len(person_list) // batch_size[0]
        self.phoneme_nbatches = len(voice_list) * 32 // batch_size[1]

        if seed is not None:
            self.reset_shuffle(seed)
        else:
            self.shuffle = np.arange(len(self))

    def __len__(self):
        return self.person_nbatches * self.phoneme_nbatches

    def reset_shuffle(self, seed=0):
        np.random.seed(seed)
        self.shuffle = np.random.permutation(len(self))

    def __getitem__(self, batch_idx):
        '''
        Returns
        -------
        data : [n_speakers, n_utterances, phoenemes_length, nfft // 2]
        '''

        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError()

        batch_idx = self.shuffle[batch_idx]

        # インデックスをperson方向とphoneme方向に分解
        phoneme_batch_idx, person_batch_idx = divmod(batch_idx, self.person_nbatches)

        # 読み込むべきファイルのpersonの範囲を求める
        person_start_idx = person_batch_idx * self.batch_size[0]
        person_end_idx   = (person_batch_idx + 1) * self.batch_size[0]

        # 読み込むべきファイルのvoiceの範囲を求める
        voice_start_idx = phoneme_batch_idx * self.batch_size[1] // 32
        voice_end_idx   = ((phoneme_batch_idx + 1) * self.batch_size[1] - 1) // 32 + 1
        
        # 複数のvoiceを結合したデータから取り出すべき範囲を求める
        voice_start_phoneme = phoneme_batch_idx * self.batch_size[1] % 32
        voice_end_phoneme   = voice_start_phoneme + self.batch_size[1]

        data = list()
        for person_idx in range(person_start_idx, person_end_idx):
            
            person_data = list()
            for voice_idx in range(voice_start_idx, voice_end_idx):
                specific = {
                    'person'     : self.person_list[person_idx] + 1,
                    'voice'      : self.voice_list[voice_idx] + 1,
                    'deform_type': self.deform_type,
                }
                pack = np.load(self.dataset_path % specific, allow_pickle=True)
                
                if self.deform_type == 'stretch':
                    sp = pack['sp']
                elif self.deform_type == 'padding':
                    sp = np.array([self._zero_padding(x[:self.phonemes_length], self.phonemes_length) for x in pack['sp']])
                elif self.deform_type == 'variable':
                    sp = np.array([x for x in pack['sp']])

                person_data.extend(sp)
            data.append(person_data[voice_start_phoneme:voice_end_phoneme])
        data = np.array(data)

        if self.mel_basis is not None:
            data = np.dot(data, self.mel_basis.T)
        else:
            data = data[:, :, 1:]
        
        data  = torch.from_numpy(data).float().to('cuda')
        return data

    @staticmethod
    def _zero_padding(x, target_length):
        y_pad = target_length - len(x)
        return np.pad(x, ((0, y_pad), (0, 0)), mode='constant') if y_pad > 0 else x
