import argparse
import datetime
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch

from emb_model import FullModel
from ge2e import GE2E


def main():
    CODE_ID  = 'emb-train'
    RUN_ID   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    WORK_DIR = os.path.join('./dest', CODE_ID, RUN_ID)
    LOG_PATH = os.path.join(WORK_DIR, 'log.txt')

    os.makedirs(WORK_DIR, exist_ok=True)
    init_logger(LOG_PATH)
    logging.info(f'Output: {WORK_DIR}')
    backup_codes(['./autovc-fork/emb_train.py', './autovc-fork/emb_model.py'], WORK_DIR)

    args = get_args()
    logging.info(args)

    # model = DVector(ninput=80, nhidden=768, noutput=256).to(args.device).train()
    model = FullModel(100).to(args.device).train()
    logging.info(model)

    dataset = Utterances(args.nspkrs, args.nuttrs, args.nsmpls, args.nsteps, args.sp_path, args.train_spkr_rate)
    cxe     = torch.nn.CrossEntropyLoss()
    ge2e    = GE2E(loss_method='softmax')
    optim   = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    patience = 0
    for step, (data, true) in enumerate(dataset):
        data = data.to(args.device)
        true = true.to(args.device)

        full_output = model(data)
        emb_output  = model.embed(data)

        cxe_loss  = cxe(full_output, true)
        ge2e_loss = ge2e(emb_output.reshape(args.nspkrs, args.nuttrs, -1))
        loss = cxe_loss + args.alpha * ge2e_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            logging.info(f'[Itreration: {step}]')
            logging.info(f'TRAIN--- Loss: {loss.item():.12f} = CXE(={cxe_loss.item():.12f}) + {args.alpha} * GE2E(={ge2e_loss.item():.12f})')

            eval_data, eval_true = dataset.get_data(eval_flag=True, nspkrs=20, nuttrs=8)
            eval_data = eval_data.to(args.device)
            eval_true = eval_true.to(args.device)

            eval_emb_output  = model.embed(eval_data)

            eval_ge2e_loss = ge2e(eval_emb_output.reshape(args.nspkrs, args.nuttrs, -1))

            logging.info(f'EVAL --- GE2E: {eval_ge2e_loss.item():.12f}')

            if eval_ge2e_loss.item() < best_loss:
                patience = 0
                best_loss = eval_ge2e_loss.item()
                logging.info(f'Improved.')
                torch.save(model.state_dict(), os.path.join(WORK_DIR, 'weights.pth'))
            else:
                patience += 1
                logging.info(f'No improvement. Patience: {patience} / {args.patience}')
                if patience >= args.patience:
                    break

    logging.info(f'Best loss: {best_loss}')

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sp_path', type=str, default='./resource/sp/phonemes_v1')
    parser.add_argument('--device',  type=str, default='cuda:0')

    parser.add_argument('--nspkrs', type=int, default=8)
    parser.add_argument('--nuttrs', type=int, default=16)
    parser.add_argument('--nsmpls', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=100000)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--train_spkr_rate', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=20)

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

    def __init__(self, nspkrs, nuttrs, nsmpls, nsteps, path, train_spkr_rate=1):
        self.nspkrs = nspkrs
        self.nuttrs = nuttrs
        self.nsmpls = nsmpls
        self.nsteps = nsteps

        _, dir_list, _ = next(os.walk(path))
        train_nspkrs = int(len(dir_list) * train_spkr_rate)
        self.train_speaker_range = range(0,            train_nspkrs)
        self.eval_speaker_range  = range(train_nspkrs, len(dir_list))

        self.data = []
        for dir_name in sorted(dir_list):
            dir_path = os.path.join(path, dir_name)
            _, _, file_list = next(os.walk(dir_path))

            uttrs = []
            for file_name in sorted(file_list):
                file_path = os.path.join(dir_path, file_name)
                uttr = np.load(file_path)

                pad = self.nsmpls - uttr.shape[0]
                if pad > 0:
                    uttr = np.concatenate([uttr, np.zeros((pad, uttr.shape[1]))])
                else:
                    uttr = uttr[:self.nsmpls]

                uttrs.append(uttr)
            self.data.append(uttrs)
        self.data = np.array(self.data)

    def __iter__(self):
        for _ in range(self.nsteps):
            yield self.get_data()

    def get_data(self, eval_flag=False, nspkrs=None, nuttrs=None):
        if nspkrs is None:
            nspkrs = self.nspkrs
        if nuttrs is None:
            nuttrs = self.nuttrs

        spk_idcs = random.sample(
            self.train_speaker_range if not eval_flag else self.eval_speaker_range,
            self.nspkrs
        )
        spks = self.data[spk_idcs]
        batch = []
        for spk in spks:
            utt_idcs = random.sample(range(len(spk)), self.nuttrs)
            utts = spk[utt_idcs]
            batch.extend(utts)
        batch = np.array(batch)
        spk_labels = np.repeat(spk_idcs, self.nuttrs)
        return torch.from_numpy(batch).float(), torch.from_numpy(spk_labels).long()

if __name__ == '__main__':
    main()
