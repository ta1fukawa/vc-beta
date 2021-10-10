import logging
import sys

import numpy as np
import torch
import tqdm

from dataloader import DataLoader
from models import *


class FullModel(torch.nn.Module):
    def __init__(self):
        self.content_encoder_conv1d = ContentEncoderConv1d()
        self.decoder_conv1d = DecoderConv1d()

    def forward(self, source_sp, target_speaker_embed):
        content_embed = self.content_encoder_conv1d(source_sp)
        pred_sp = self.decoder_conv1d(target_speaker_embed, content_embed)
        return pred_sp

def main():
    init_logger('happy.log')

    model = FullModel()
    learn_loader = DataLoader()
    eval_loader = DataLoader()
    history = train(model, (learn_loader, eval_loader), f'weights.pth', 1e-3, 6)
    
    pass

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
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

    best_loss = np.Inf
    wait = 0
    history = { key: list() for key in ['train_loss', 'train_acc', 'valid_loss', 'valid_acc'] }
    for epoch in range(256):
        logging.info(f'[Epoch {epoch}]')

        train_loss, train_acc = learn(model, loaders[0], optimizer, criterion)
        logging.info(f'Train loss {train_loss}, acc {100 * train_acc} %')

        valid_loss, valid_acc = valid(model, loaders[1], criterion)
        logging.info(f'Valid loss {valid_loss}, acc {100 * valid_acc} %')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

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
    train_acc  = 0
    with tqdm.tqdm(loader, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for idx, batch in enumerate(bar):
            data, true = batch
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += pred.argmax(dim=1).eq(true).sum().item()
            bar.set_postfix({'loss': '%.4f' % (train_loss / (idx + 1)), 'acc': '%.2f %%' % (100 * train_acc / ((idx + 1) * len(true)))})

    train_loss /= len(loader) * len(true)
    train_acc  /= len(loader) * len(true)
    return train_loss, train_acc

def valid(model, loader, criterion):
    model.eval()

    valid_loss = 0
    valid_acc  = 0
    for batch in loader:
        with torch.no_grad():
            data, true = batch
            pred = model(data)
            loss = criterion(pred, true)

            valid_loss += loss.item()
            valid_acc  += pred.argmax(dim=1).eq(true).sum().item()

    valid_loss /= len(loader) * len(true)
    valid_acc  /= len(loader) * len(true)
    return valid_loss, valid_acc

if __name__ == '__main__':
    main()
