import json
import logging
import sys

import numpy as np
import torch
import tqdm

from dataloader import DataLoader
from models import *


class FullModel(torch.nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()

        self.content_encoder_conv1d = ContentEncoderConv1d()
        self.decoder_conv1d = DecoderConv1d()

    def forward(self, source_sp, target_speaker_embed):
        content_embed = self.content_encoder_conv1d(source_sp)
        pred_sp = self.decoder_conv1d(target_speaker_embed, content_embed)
        return pred_sp

def main():
    init_logger('happy.log')

    model = FullModel().to(f'cuda')
    learn_loader = DataLoader([10], range(12), range(12), batch_size=256, sp_length=1024, preload=True)
    eval_loader = DataLoader([10], range(12, 16), range(12, 16), batch_size=32, sp_length=1024, preload=True)
    history = train(model, (learn_loader, eval_loader), f'weights.pth', 1e-3, 6)
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

        train_loss = learn(model, loaders[0], optimizer, criterion)
        logging.info(f'Train loss {train_loss}')

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
            optimizer.zero_grad()
            pred = model(data, spkr)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix({'loss': '%.8f' % (train_loss / (idx + 1))})

    train_loss /= len(loader) * len(true)
    return train_loss

def valid(model, loader, criterion):
    model.eval()

    valid_loss = 0
    for batch in loader:
        with torch.no_grad():
            data, true = batch
            pred = model(data)
            loss = criterion(pred, true)

            valid_loss += loss.item()

    valid_loss /= len(loader) * len(true)
    return valid_loss

if __name__ == '__main__':
    main()
