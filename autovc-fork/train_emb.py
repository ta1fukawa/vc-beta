import os
import random

import numpy as np
import torch

from ge2e import GE2E
from model import XVectorConv2D


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

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1] // (self.nspkrs * self.nuttrs)

    def __iter__(self):
        for _ in range(self.nsteps):
            spks = random.sample(self.data, self.nspkrs)
            batch = []
            for spk in spks:
                utts = random.sample(spk, self.nuttrs)
                batch.append(utts)
            yield torch.from_numpy(np.array(batch)).float()

def main():
    device = 'cuda:1'
    nsteps = 100000

    nspkrs   = 8
    nuttrs   = 8
    nsamples = 512

    # model = DVector(ninput=80, nhidden=768, noutput=256).to(device).train()
    model = XVectorConv2D().to(device).train()
    print(model)

    dataset = Utterances(nspkrs, nuttrs, nsamples, nsteps)
    criterion = GE2E(loss_method='softmax')
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step, data in enumerate(dataset):
        data = data.to(device)
        y = model(data.reshape(nspkrs * nuttrs, nsamples, -1)).reshape(nspkrs, nuttrs, -1)

        loss = criterion(y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 100 == 0:
            print(f'Iteration: {step}, Loss: {loss.item()}')
            torch.save(model.state_dict(), './dest/test-03/weights.pth')

if __name__ == '__main__':
    main()
