import glob

import numpy as np
import torch

from dataloader import DataLoader
from model import *

model = DVectorModel().to('cuda')
model.load_state_dict(torch.load(sorted(glob.glob('dest/test-02/*/weights.pth'))[-1]))
model.eval()

embeds = list()
for speaker in range(100):
    loader = DataLoader([10], [speaker], range(100), (1, 25), sp_length=1024, preload=False, mel=False)

    pred_list = list()
    for data in loader:
        with torch.no_grad():
            pred = model(data).squeeze(0).squeeze(0)
        pred_list.extend(pred.to('cpu').detach().numpy().copy())
    embeds.append(np.array(pred_list))

centroids = np.mean(embeds, axis=1)
np.savez_compressed('resource/emb/emb2_centroids.npz', centroids=centroids)
