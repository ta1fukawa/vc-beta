import os
import random

import numpy as np
import torch

from model import AutoVC


class Utterances(object):

    def __init__(self, nitems, nsamples, nsteps):
        self.nitems   = nitems
        self.nsamples = nsamples
        self.nsteps   = nsteps

        path = './resource/seiren_jvs011_sp'
        _, dir_list, _ = next(os.walk(path))

        self.data = []
        for dir_name in sorted(dir_list):
            dir_path = os.path.join(path, dir_name)
            _, _, file_list = next(os.walk(dir_path))
            
            uttrs = []
            for file_name in sorted(file_list):
                file_path = os.path.join(dir_path, file_name)
                uttr = np.load(file_path)

                pad = self.nsamples - uttr.shape[0]
                if pad > 0:
                    uttr = np.concatenate([uttr, np.zeros((pad, uttr.shape[1]))])
                else:
                    uttr = uttr[:self.nsamples]

                uttrs.append(uttr)
            self.data.append(uttrs)

        self.data = np.array(self.data)
        self.embs = np.load('./resource/emb/emb3_centroids.npy', allow_pickle=True)

    def __iter__(self):
        for _ in range(self.nsteps):
            spk_idcs  = random.sample(range(len(self.data)), self.nitems * 2)
            uttr_idcs = random.sample(range(len(self.data[0])), self.nitems)
            src_spk_idcs = spk_idcs[:self.nitems]
            tgt_spk_idcs = spk_idcs[self.nitems:]

            src_uttrs = torch.from_numpy(np.array(self.data[src_spk_idcs, uttr_idcs])).float()
            src_embs  = torch.from_numpy(np.array(self.embs[src_spk_idcs])).float()
            tgt_uttrs = torch.from_numpy(np.array(self.data[tgt_spk_idcs, uttr_idcs])).float()
            tgt_embs  = torch.from_numpy(np.array(self.embs[tgt_spk_idcs])).float()

            yield (src_uttrs, src_embs, tgt_uttrs, tgt_embs)

def main():
    device = 'cuda:1'
    nsteps = 100000

    nitems   = 10
    nsamples = 512

    model = AutoVC(16, 16).to(device).train()
    print(model)

    dataset = Utterances(nitems, nsamples, nsteps)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    best_mean_loss = float('inf')
    patience = 0
    for step, (src_uttrs, src_embs, tgt_uttrs, tgt_embs) in enumerate(dataset):
        src_uttrs = src_uttrs.to(device)
        src_embs  = src_embs.to(device)
        # tgt_uttrs = tgt_uttrs.to(device)
        # tgt_embs  = tgt_embs.to(device)

        y_uttr, y_psnt, code_real = model(src_uttrs, src_embs, src_embs)
        
        loss_uttr = torch.nn.functional.mse_loss(y_uttr, src_uttrs)
        loss_psnt = torch.nn.functional.mse_loss(y_psnt, src_uttrs)

        code_reconst = model.encoder(y_psnt, src_embs)
        loss_cd = torch.nn.functional.l1_loss(code_real, code_reconst)

        loss = loss_uttr + loss_psnt + loss_cd
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if np.mean(losses[-100:]) < best_mean_loss:
            best_mean_loss = np.mean(losses[-100:])
            torch.save(model.state_dict(), './dest/test-04/weights.pth')
            patience = 0
        else:
            patience += 1
            if patience >= 100:
                break

        if step % 100 == 0:
            print(f'Iteration: {step}, Loss: {loss.item()}')

if __name__ == '__main__':
    main()
