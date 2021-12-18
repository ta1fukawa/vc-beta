import torch
import numpy as np

class MelEmbLoader(torch.utils.data.Dataset):
    def __init__(self, mel_dir, embed_dir, embed_dim, seg_len):
        self.mel_dir = mel_dir
        self.embed_dir = embed_dir
        self.embed_dim = embed_dim
        self.seg_len   = seg_len

        self.mel_files   = sorted(list(mel_dir.glob('*/*.pt')))
        self.embed_files = [embed_dir / f'{f.parent.name}.pt' for f in self.mel_files]

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        src_mel = torch.load(self.mel_files[idx])
        src_emb = torch.load(self.embed_files[idx])

        if len(src_mel) < self.seg_len:
            len_pad = self.seg_len - len(src_mel)
            src_mel = torch.cat((src_mel, torch.zeros(len_pad, self.embed_dim)))
        else:
            src_mel = src_mel[:self.seg_len]

        return src_mel, src_emb