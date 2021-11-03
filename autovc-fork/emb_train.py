import datetime
import logging
import os
import random
import sys

import numpy as np
import torch

from emb_model import FullModel
from ge2e import GE2E


device = 'cuda:1'
nsteps = 100000

nspkrs   = 8
nuttrs   = 8
nsamples = 512

alpha = 0.1

class Utterances(object):

    def __init__(self, nspkrs, nuttrs, nsamples, nsteps):
        self.nspkrs   = nspkrs
        self.nuttrs   = nuttrs
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

    def __iter__(self):
        for _ in range(self.nsteps):
            spk_idcs = random.sample(range(len(self.data)), self.nspkrs)
            spks = self.data[spk_idcs]
            batch = []
            for spk in spks:
                utt_idcs = random.sample(range(len(spk)), self.nuttrs)
                utts = spk[utt_idcs]
                batch.extend(utts)
            batch = np.array(batch)
            spk_labels = np.repeat(spk_idcs, self.nuttrs)
            yield torch.from_numpy(batch).float(), torch.from_numpy(spk_labels).long()

def main():
    code_id  = 'emb-train'
    run_id   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = os.path.join('./dest', code_id, run_id)
    os.makedirs(work_dir, exist_ok=True)
    init_logger(os.path.join(work_dir, 'general.log'))

    # model = DVector(ninput=80, nhidden=768, noutput=256).to(device).train()
    model = FullModel(100).to(device).train()
    logging.info(model)

    dataset = Utterances(nspkrs, nuttrs, nsamples, nsteps)
    cxe     = torch.nn.CrossEntropyLoss()
    ge2e    = GE2E(loss_method='softmax')
    optim   = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    best_mean_loss = float('inf')
    best_loss = float('inf')
    patience = 0
    for step, (data, true) in enumerate(dataset):
        data = data.to(device)
        true = true.to(device)

        full_output = model(data)
        emb_output  = model.embed(data)

        cxe_loss  = cxe(full_output, true)
        ge2e_loss = ge2e(emb_output.reshape(nspkrs, nuttrs, -1))
        loss = cxe_loss + alpha * ge2e_loss
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            logging.info(f'Iteration: {step}, Loss: {loss.item()} (CXE: {cxe_loss.item()}, GE2E: {ge2e_loss.item()})')

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
