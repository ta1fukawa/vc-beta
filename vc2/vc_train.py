import argparse
import datetime
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch

from vc_model import AutoVC, AutoVCConv2d


def main():
    CODE_ID  = 'vc-train'
    RUN_ID   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    WORK_DIR = os.path.join('./dest', CODE_ID, RUN_ID)
    LOG_PATH = os.path.join(WORK_DIR, 'log.txt')

    os.makedirs(WORK_DIR, exist_ok=True)
    init_logger(LOG_PATH)
    logging.info(f'Output: {WORK_DIR}')
    backup_codes(['./vc2/vc_train.py', './vc2/vc_model.py'], WORK_DIR)

    args = get_args()
    logging.info(args)

    model = AutoVC(args.dim_neck, args.skip_len).to(args.device).train()
    # model = AutoVCConv2d(args.emb_dims, args.nsamples, args.nmels, args.nlayers, args.postnet_nlayers, args.nchannels).to(args.device).train()
    logging.info(model)

    dataset = Utterances(args.nitems, args.nsamples, args.nsteps, args.emb_path)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    best_mean_loss = float('inf')
    best_loss = float('inf')
    patience = 0
    for step, (src_uttrs, src_embs, tgt_uttrs, tgt_embs) in enumerate(dataset):
        src_uttrs = src_uttrs.to(args.device)
        src_embs  = src_embs.to(args.device)
        if args.same_flag:
            tgt_uttrs = src_uttrs
            tgt_embs  = src_embs
        else:
            tgt_uttrs = tgt_uttrs.to(args.device)
            tgt_embs  = tgt_embs.to(args.device)

        y_uttr, y_psnt, code_real = model(src_uttrs, src_embs, tgt_embs)

        loss_uttr = torch.nn.functional.mse_loss(y_uttr, tgt_uttrs)
        loss_psnt = torch.nn.functional.mse_loss(y_psnt, tgt_uttrs)

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
                torch.save(model.state_dict(), os.path.join(WORK_DIR, 'weights.pth'))

            if np.mean(losses[-10:]) < best_mean_loss:
                best_mean_loss = np.mean(losses[-10:])
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

    logging.info(f'Best mean loss: {best_mean_loss}')

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_path', type=str, default='./dest/emb-main/20211104-000810/centroids.npy')
    parser.add_argument('--device',   type=str, default='cuda:0')

    parser.add_argument('--nitems',   type=int, default=8)
    parser.add_argument('--nsamples', type=int, default=512)
    parser.add_argument('--nmels',    type=int, default=80)
    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--nsteps',   type=int, default=100000)
    parser.add_argument('--nlayers',  type=int, default=3)
    parser.add_argument('--postnet_nlayers', type=int, default=5)
    parser.add_argument('--nchannels',type=int, default=128)
    parser.add_argument('--dim_neck', type=int, default=64)
    parser.add_argument('--skip_len', type=int, default=4)

    parser.add_argument('--same_flag', type=bool, default=False)

    args = parser.parse_args()
    return args

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def backup_codes(src_files, dest_dir):
    for src_file in src_files:
        shutil.copyfile(src_file, os.path.join(dest_dir, os.path.split(src_file)[1]))

class Utterances(object):

    def __init__(self, nitems, nsamples, nsteps, emb_path):
        self.nitems   = nitems
        self.nsamples = nsamples
        self.nsteps   = nsteps

        path = './resource/mel/utterances'
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
        self.embs = np.load(emb_path, allow_pickle=True)

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

if __name__ == '__main__':
    main()
