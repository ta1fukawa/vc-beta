import glob

import numpy as np
import torch

from dataloader import DataLoader
from model import SpeakerEmbedModel

person_list = np.arange(100)
voice_list  = np.arange(26)
dataset_path = '../vc-alpha/resource/jvs_ver1/data_32_16/jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz'
loader = DataLoader(person_list, voice_list, (100, 2), dataset_path, 'stretch', 32, False)

model = SpeakerEmbedModel().to('cuda')
model.load_state_dict(torch.load(sorted(glob.glob('dest/default_ge2e/stretch/*/weights.pth'))[-1]))
model.eval()

pred_list = list()
for data in loader:
    with torch.no_grad():
        pred = model(data)
    pred_list.append(pred.to('cpu').detach().numpy().copy())
pred = np.reshape(np.transpose(pred_list, (1, 0, 2, 3)), (len(loader[0]), -1, pred.shape[-1]))

centroids = np.mean(pred, axis=1)
np.savez_compressed('resource/emb/ge2e_centroids.npz', centroids=centroids)
