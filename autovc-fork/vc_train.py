import datetime
import logging
import os
import random
import sys

import numpy as np
import torch

from vc_model import AutoVCConv2d


centroids_path = './dest/emb-main/20211104-000810/centroids.npy'

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
        self.embs = np.load(centroids_path, allow_pickle=True)

    def __iter__(self):
        for _ in range(self.nsteps):
            spk_idcs = random.sample(range(len(self.data)), self.nitems * 2)
            src_spk_idcs = spk_idcs[:self.nitems]
            tgt_spk_idcs = spk_idcs[self.nitems:]
            utt_idcs = random.sample(range(len(self.data[0])), self.nitems)

            src_utts = torch.from_numpy(self.data[src_spk_idcs, utt_idcs]).float()
            src_embs = torch.from_numpy(np.array(self.embs[src_spk_idcs])).float()
            tgt_utts = torch.from_numpy(np.array(self.data[tgt_spk_idcs, utt_idcs])).float()
            tgt_embs = torch.from_numpy(np.array(self.embs[tgt_spk_idcs])).float()

            yield (src_utts, src_embs, tgt_utts, tgt_embs)

def main():
    code_id  = 'vc-train'
    run_id   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = os.path.join('./dest', code_id, run_id)
    os.makedirs(work_dir, exist_ok=True)
    init_logger(os.path.join(work_dir, 'general.log'))

    device = 'cuda:1'
    nsteps = 100000

    nitems   = 8
    nsamples = 512
    nmels    = 80
    emb_dims = 512

    # model = AutoVC(16, 16).to(device).train()
    model = AutoVCConv2d(emb_dims, nsamples, nmels).to(device).train()
    logging.info(model)

    dataset = Utterances(nitems, nsamples, nsteps)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    best_mean_loss = float('inf')
    best_loss = float('inf')
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

        if step % 100 == 0:
            logging.info(f'Iteration: {step}, Loss: {loss.item()} (uttr: {loss_uttr.item()}, psnt: {loss_psnt.item()}, cd: {loss_cd.item()})')

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(work_dir, 'weights.pth'))

            if np.mean(losses[-10:]) < best_mean_loss:
                best_mean_loss = np.mean(losses[-10:])
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

    logging.info(f'Best mean loss: {best_mean_loss}')

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    main()
