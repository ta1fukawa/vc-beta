import csv
import datetime
import glob
import json
import logging
import os
import sys

import numpy as np
import torch
import tqdm
from sklearn import multiclass, svm

from dataloader import *
from ge2e import GE2ELoss
from model import *

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f'dest/test-02/{run_id}'
    os.makedirs(output_dir, exist_ok=True)
    init_logger(os.path.join(output_dir, 'general.log'))

    person_known_size   = 80
    person_unknown_size = 20
    voice_train_size    = 80
    voice_check_size    = 20
    seiren_speaker_list = [10]

    mel = False

    known_person_list   = np.arange(person_known_size)
    unknown_person_list = np.arange(person_known_size, person_known_size + person_unknown_size)
    train_voice_list = np.arange(voice_train_size)
    check_voice_list = np.arange(voice_train_size, voice_train_size + voice_check_size)

    model = DVectorModel().to('cuda')
    # learn_loader = DataLoader(seiren_speaker_list, known_person_list,   train_voice_list, batch_size=(4, 4), sp_length=1024, preload=False, mel=mel)
    # eval_loader  = DataLoader(seiren_speaker_list, unknown_person_list, check_voice_list, batch_size=(4, 4), sp_length=1024, preload=False, mel=mel)
    # history = train(model, (learn_loader, eval_loader), f'dest/test-02/{run_id}/weights.pth', 1e-3, 6)
    # logging.info('History:\n' + json.dumps(history, ensure_ascii=False, indent=4))
    load_weights(model, 'dest/test-02/*/weights.pth')
    
    logging.info('Start evaluation')
    known_train_loader   = DataLoader(seiren_speaker_list, known_person_list[:20], train_voice_list, (4, 4), sp_length=1024, preload=True, mel=mel)
    known_eval_loader    = DataLoader(seiren_speaker_list, known_person_list[:20], check_voice_list, (4, 4), sp_length=1024, preload=True, mel=mel)
    unknown_train_loader = DataLoader(seiren_speaker_list, unknown_person_list,    train_voice_list, (4, 4), sp_length=1024, preload=True, mel=mel)
    unknown_eval_loader  = DataLoader(seiren_speaker_list, unknown_person_list,    check_voice_list, (4, 4), sp_length=1024, preload=True, mel=mel)
    known_train_embed_pred   = pred_(model, known_train_loader)
    known_eval_embed_pred    = pred_(model, known_eval_loader)
    unknown_train_embed_pred = pred_(model, unknown_train_loader)
    unknown_eval_embed_pred  = pred_(model, unknown_eval_loader)
    known_svm_confusion_matrix  , n_known_corrects,   n_known_data   = calc_svm(known_train_embed_pred,   known_eval_embed_pred)
    unknown_svm_confusion_matrix, n_unknown_corrects, n_unknown_data = calc_svm(unknown_train_embed_pred, unknown_eval_embed_pred)
    logging.info(f'Known accuracy: {n_known_corrects / n_known_data} ({n_known_corrects}/{n_known_data})')
    logging.info(f'Unknown accuracy: {n_unknown_corrects / n_unknown_data} ({n_unknown_corrects}/{n_unknown_data})')
    with open(os.path.join(output_dir, 'known_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(known_svm_confusion_matrix)
    with open(os.path.join(output_dir, 'unknown_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(unknown_svm_confusion_matrix)

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def load_weights(model, weights_path):
    existing_weights_paths = sorted(glob.glob(weights_path))
    if len(existing_weights_paths) == 0:
        logging.info('Weights is not found.')
        return

    logging.info('Loading weights: ' + existing_weights_paths[-1])
    model.load_state_dict(torch.load(existing_weights_paths[-1]))

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
    
def calc_confusion_matrix(pred, true, nclasses):
    confusion_matrix = np.zeros((nclasses, nclasses))
    for eval_true_label, pred_label in zip(true, pred):
        confusion_matrix[eval_true_label][pred_label] += 1
    return confusion_matrix

def calc_svm(train_data, eval_data):
    logging.info('Start svm learning')
    n_classes, n_train_items, embed_dim = train_data.shape
    n_classes, n_eval_items,  embed_dim = eval_data.shape

    train_label = np.concatenate([[person_idx] * n_train_items for person_idx in range(n_classes)])
    svc = svm.SVC(C=1., kernel='rbf')
    classifier = multiclass.OneVsRestClassifier(svc)
    classifier.fit(train_data.reshape(-1, embed_dim), train_label)

    eval_pred  = classifier.predict(eval_data.reshape(-1, embed_dim))
    eval_label = np.concatenate([[person_idx] * n_eval_items for person_idx in range(n_classes)])
    confusion_matrix = calc_confusion_matrix(eval_pred, eval_label, n_classes)

    n_corrects = np.trace(confusion_matrix).astype(np.int)
    n_data = (n_classes * n_eval_items)
    return confusion_matrix, n_corrects, n_data

if __name__ == '__main__':
    main()
