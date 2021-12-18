import torch
import numpy as np

class MelEmbLoader(torch.utils.data.Dataset):
    def __init__(self, mel_dir, embed_dir, embed_dim):
        self.mel_dir = mel_dir
        self.embed_dir = embed_dir
        self.embed_dim = embed_dim

        self.mel_files = list(self.mel_dir.glob('*.npy'))
        self.embed_files = list(self.embed_dir.glob('*.npy'))

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        src_mel = torch.from_numpy(np.load(self.mel_files[idx])).float()
        src_emb = torch.from_numpy(np.load(self.embed_files[idx])).float()
        return src_mel, src_emb