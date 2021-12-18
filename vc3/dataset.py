import torch

import common

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

        src_mel = common.pad_seq(src_mel, self.seg_len)

        return src_mel, src_emb