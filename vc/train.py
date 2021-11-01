import json
import logging
import sys
import os

import numpy as np
import torch
import tqdm
import datetime

from dataloader import *
from models import *


class FullModel(torch.nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()

        self.content_encoder = ContentEncoderConv2d()
        self.decoder = DecoderConv2d()

    def forward(self, source_sp, target_speaker_embed):
        content_embed = self.content_encoder(source_sp)
        pred_sp = self.decoder(target_speaker_embed, content_embed, self.content_encoder.pool_indices)
        return pred_sp

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.first_conv = torch.nn.Conv2d(1, 8, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.first_line = torch.nn.Linear(512, 8 * 512)

        self.last_conv = torch.nn.Conv2d(8 + 8, 1, kernel_size=(3, 3), dilation=(1, 1), padding='same')

    def forward(self, source_sp, target_speaker_embed):
        source_sp = torch.unsqueeze(source_sp, 1)
        source_sp = self.first_conv(source_sp)

        target_speaker_embed = self.first_line(target_speaker_embed)
        target_speaker_embed = torch.reshape(target_speaker_embed, (-1, 8, 1, source_sp.shape[3]))
        target_speaker_embed = target_speaker_embed.expand(-1, -1, source_sp.shape[2], -1)
        x = torch.cat((source_sp, target_speaker_embed), dim=1)
        x = self.last_conv(x)
        return x

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f'dest/test-01/{run_id}', exist_ok=True)
    init_logger(f'dest/test-01/{run_id}/general.log')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = FullModel().to('cuda')
    learn_loader = DataLoaderVary([10], range(64),     range(16),     batch_size=1, sp_length=1024, preload=True)
    eval_loader  = DataLoaderVary([10], range(64, 80), range(16, 20), batch_size=1, sp_length=1024, preload=True)
    history = train(model, (learn_loader, eval_loader), f'dest/test-01/{run_id}/weights.pth', 1e-3, 6)
    logging.info('History:\n' + json.dumps(history, ensure_ascii=False, indent=4))

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def train(model, loaders, weights_path, leaning_rate, patience):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

    best_loss = np.Inf
    wait = 0
    history = { key: list() for key in ['train_loss', 'valid_loss'] }
    for epoch in range(256):
        logging.info(f'[Epoch {epoch}]')

        loaders[0].set_seed(epoch)
        train_loss = learn(model, loaders[0], optimizer, criterion)
        logging.info(f'Train loss {train_loss}')

        loaders[1].set_seed(epoch)
        valid_loss = valid(model, loaders[1], criterion)
        logging.info(f'Valid loss {valid_loss}')

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_loss:
            wait = 0
            best_loss = valid_loss
            logging.info(f'val_loss improved.')
            torch.save(model.state_dict(), weights_path)
        else:
            wait += 1
            logging.info(f'val_loss did not improve. {wait}/{patience}')
            if wait >= patience:
                logging.info(f'Early stopping.')
                model.load_state_dict(torch.load(weights_path))
                break
    return history

def learn(model, loader, optimizer, criterion):
    model.train()

    train_loss = 0
    with tqdm.tqdm(loader, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for idx, batch in enumerate(bar):
            data, true, spkr = batch
            pred = model(data, spkr)
            loss = criterion(pred, true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix({'loss': '%.4e' % (train_loss / (idx + 1))})

    train_loss /= len(loader) * len(true)
    return train_loss

def valid(model, loader, criterion):
    model.eval()

    valid_loss = 0
    for batch in loader:
        with torch.no_grad():
            data, true, spkr = batch
            pred = model(data, spkr)
            loss = criterion(pred, true)

            valid_loss += loss.item()

    valid_loss /= len(loader) * len(true)
    return valid_loss

if __name__ == '__main__':
    main()
