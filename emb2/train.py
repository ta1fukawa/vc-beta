import json
import logging
import sys
import os

import numpy as np
import torch
import tqdm
import datetime

from dataloader import *
from model import *
from ge2e import GE2ELoss

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f'dest/test-02/{run_id}', exist_ok=True)
    init_logger(f'dest/test-02/{run_id}/general.log')

    mel = False

    model = DVectorModel(dim_input=(128 if mel else 512), dim_cell=768, dim_emb=256).to('cuda')
    learn_loader = DataLoader([10], range(80),      range(80),      batch_size=(4, 4), sp_length=1024, preload=False, mel=mel)
    eval_loader  = DataLoader([10], range(80, 100), range(80, 100), batch_size=(4, 4), sp_length=1024, preload=False, mel=mel)
    history = train(model, (learn_loader, eval_loader), f'dest/test-02/{run_id}/weights.pth', 1e-3, 6)
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
    criterion = GE2ELoss(loss_method='softmax')
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

    batch_size = len(loader[0]) * len(loader[0][0])
    train_loss = 0
    with tqdm.tqdm(loader, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for idx, data in enumerate(bar):
            optimizer.zero_grad()
            pred = model(data)
            pred = torch.reshape(pred, (-1, 4, 256))
            loss = criterion(pred)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.set_postfix({'loss': '%.4f' % (train_loss / ((idx + 1) * batch_size))})

    train_loss /= len(loader) * batch_size
    return train_loss

def valid(model, loader, criterion):
    model.eval()

    batch_size = len(loader[0]) * len(loader[0][0])
    valid_loss = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            pred = torch.reshape(pred, (-1, 4, 256))
            loss = criterion(pred)

            valid_loss += loss.item()

    valid_loss /= len(loader) * batch_size
    return valid_loss

def pred_(model, loader):
    model.eval()

    pred_list = list()
    for data in loader:
        with torch.no_grad():
            pred = model(data)
        pred = torch.reshape(pred, (-1, 4, 256))
        pred_list.append(pred.to('cpu').detach().numpy().copy())
    pred = np.reshape(np.transpose(pred_list, (1, 0, 2, 3)), (len(loader[0]), -1, pred.shape[-1]))
    return pred
    
if __name__ == '__main__':
    main()
