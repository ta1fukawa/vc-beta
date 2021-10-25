import argparse
import datetime
import glob
import logging
import os
import shutil
import sys
import json
import warnings
import csv

import easydict
import numpy as np
import torch
import tqdm
from sklearn import multiclass, svm

from dataloader import DataLoader
from model import XvecModel
from omake import console_menu, console_inputarea
from ge2e import GE2ELoss

def get_args():
    parser = argparse.ArgumentParser(description='研究用：音素に対して時間領域で処理して話者埋め込みを求めるやつ')
    parser.add_argument('--gpu',      default=None,           type=str, metavar='N',    help='GPU番号')
    parser.add_argument('--dest-dir', default='dest',         type=str, metavar='PATH', help='出力先ディレクトリのパス')
    parser.add_argument('--code-id',  default='default_xvec', type=str, metavar='ID',   help='プログラムコードの識別コード')
    parser.add_argument('--no-backup', action='store_true', help='Pythonコードのバックアップの無効化（推奨）')

    parser.add_argument('--dataset-dir', default='../vc-alpha/resource/jvs_ver1/data_32_16', type=str, metavar='PATH', help='データセットの書式付きパス')
    
    parser.add_argument('--patience', default=4, type=int, metavar='N', help='Early Stoppingまでの回数')
    
    parser.add_argument('-nw', '--no-load-weights', action='store_true', help='重み読み込みの有無')
    parser.add_argument('-nl', '--no-learn',        action='store_true', help='学習の有無')
    parser.add_argument(       '--use-mel',         action='store_true', help='メルスペクトログラムの使用の有無')

    parser.add_argument('--sampling-rate', default=24000, type=int, metavar='N', help='サンプリング周波数')
    parser.add_argument('--nfft',          default=1024,  type=int, metavar='N', help='STFTのウィンドウ幅（通常はPyWorldに依存）')
    parser.add_argument('--nhop',          default=120,   type=int, metavar='N', help='STFTのシフト幅（通常はPyWorldに依存）')

    parser.add_argument('-d', '--deform-type', default=None, type=str, metavar='TYPE', help='変形の種類（variableの場合はバッチサイズを(1, 1)にする）')

    parser.add_argument('-bp', '--batch-length-phoneme', default=4,  type=int, metavar='N', help='各バッチの音素数')
    parser.add_argument('-pl', '--phonemes-length',      default=32, type=int, metavar='N', help='音素の時間長')

    parser.add_argument('-pk', '--person-known-size',   default=64, type=int, metavar='N', help='既知の話者として使用する話者数')
    parser.add_argument('-pu', '--person-unknown-size', default=16, type=int, metavar='N', help='未知の話者として使用する話者数')
    parser.add_argument('-vt', '--voice-train-size',    default=20, type=int, metavar='N', help='学習に使用する音声ファイル数')
    parser.add_argument('-vc', '--voice-check-size',    default=6,  type=int, metavar='N', help='検証に使用する音声ファイル数')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    return args

def backup_code(targets, dest_dir):
    for target in targets:
        code_files = sorted(glob.glob(target))
        for code_file in code_files:
            shutil.copyfile(code_file, os.path.join(dest_dir, os.path.split(code_file)[1]))

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def main(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # コードのバックアップ（実行結果とコードの結びつきを管理するための臨時措置）
    if not cfg.no_backup:
        backup_code(['python/*.py'], cfg.output_dir)

    # ログ
    init_logger(os.path.join(cfg.output_dir, 'general.log'))

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    warnings.filterwarnings('ignore')
    logging.debug('Config:\n' + json.dumps(args, ensure_ascii=False, indent=4))

    # known_person_list   = [89, 90, 39, 2, 27, 68, 24, 99, 40, 34, 61, 71, 26, 63, 9, 92]
    # unknown_person_list = [94, 64, 0, 13, 77, 7, 16, 17, 57, 65, 84, 15, 46, 48, 70, 56]
    known_person_list   = np.arange(cfg.person_known_size)
    unknown_person_list = np.arange(cfg.person_known_size, cfg.person_known_size + cfg.person_unknown_size)
    train_voice_list = np.arange(cfg.voice_train_size)
    check_voice_list = np.arange(cfg.voice_train_size, cfg.voice_train_size + cfg.voice_check_size)  # max: 26

    logging.debug('known_person_list: '   + str(known_person_list))
    logging.debug('unknown_person_list: ' + str(unknown_person_list))
    logging.debug('train_voice_list: '    + str(train_voice_list))
    logging.debug('check_voice_list: '    + str(check_voice_list))

    model = XvecModel(cfg.person_known_size).to('cuda')
    logging.info('Model:\n' + str(model))

    if not cfg.no_load_weights:
        load_weights(model, os.path.join(cfg.output_wild_dir, 'weights.pth'))

    if not cfg.no_learn:
        weights_path = os.path.join(cfg.output_dir, 'weights.pth')
        logging.info('Start learning: ' + weights_path)

        train_loader = DataLoader(known_person_list, train_voice_list, (cfg.person_known_size, cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel, seed=0)
        valid_loader = DataLoader(known_person_list, check_voice_list, (cfg.person_known_size, cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel, seed=0)
        unknown_valid_loader = DataLoader(unknown_person_list, check_voice_list, (cfg.person_unknown_size, cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel, seed=0)
        history = learn(model, (train_loader, valid_loader, unknown_valid_loader), weights_path, leaning_rate=1e-3, patience=cfg.patience)
        logging.info('History:\n' + json.dumps(history, ensure_ascii=False, indent=4))
        
        try:
            logging.info(str([metrics for metrics in history.keys()]) + ': ' + str([metrics[-(cfg.patience + 1)] for metrics in history.values()]))
        except IndexError:
            logging.info('It could not be learned.')

    logging.info('Start evaluation')
    model.embed.reshape_flag = True
    known_train_loader   = DataLoader(known_person_list,   train_voice_list, (cfg.person_known_size,   cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel)
    known_eval_loader    = DataLoader(known_person_list,   check_voice_list, (cfg.person_known_size,   cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel)
    unknown_train_loader = DataLoader(unknown_person_list, train_voice_list, (cfg.person_unknown_size, cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel)
    unknown_eval_loader  = DataLoader(unknown_person_list, check_voice_list, (cfg.person_unknown_size, cfg.batch_length_phoneme), cfg.dataset_path, cfg.deform_type, cfg.phonemes_length, cfg.use_mel)
    known_train_embed_pred   = pred_(model.embed, known_train_loader)
    known_eval_embed_pred    = pred_(model.embed, known_eval_loader)
    unknown_train_embed_pred = pred_(model.embed, unknown_train_loader)
    unknown_eval_embed_pred  = pred_(model.embed, unknown_eval_loader)
    known_svm_confusion_matrix  , n_known_corrects,   n_known_data   = calc_svm(known_train_embed_pred,   known_eval_embed_pred)
    unknown_svm_confusion_matrix, n_unknown_corrects, n_unknown_data = calc_svm(unknown_train_embed_pred, unknown_eval_embed_pred)
    logging.info(f'Known accuracy: {n_known_corrects / n_known_data} ({n_known_corrects}/{n_known_data})')
    logging.info(f'Unknown accuracy: {n_unknown_corrects / n_unknown_data} ({n_unknown_corrects}/{n_unknown_data})')
    with open(os.path.join(cfg.output_dir, 'known_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(known_svm_confusion_matrix)
    with open(os.path.join(cfg.output_dir, 'unknown_svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(unknown_svm_confusion_matrix)

def load_weights(model, weights_path):
    existing_weights_paths = sorted(glob.glob(weights_path))
    if len(existing_weights_paths) == 0:
        logging.info('Weights is not found.')
        return

    logging.info('Loading weights: ' + existing_weights_paths[-1])
    model.load_state_dict(torch.load(existing_weights_paths[-1]))

def learn(model, loaders, weights_path, leaning_rate, patience):
    criterion = torch.nn.NLLLoss()
    metric    = GE2ELoss(loss_method='softmax')
    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

    best_loss = np.Inf
    wait = 0
    history = { key: list() for key in ['train_loss', 'train_acc', 'train_metr', 'valid_loss', 'valid_acc', 'valid_metr', 'unknown_valid_loss'] }
    for epoch in range(256):
        logging.info(f'[Epoch {epoch}]')

        train_loss, train_acc, train_metr = train(model, loaders[0], optimizer, criterion, metric)
        logging.info(f'Train loss {train_loss}, acc {100 * train_acc} %, ge2e {train_metr}')

        valid_loss, valid_acc, valid_metr = valid(model, loaders[1], criterion, metric)
        logging.info(f'Valid loss {valid_loss}, acc {100 * valid_acc} %, ge2e {valid_metr}')

        unknown_valid_loss = unknown_valid(model.embed, loaders[2], metric)
        logging.info(f'Unknown valid ge2e {unknown_valid_loss}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_metr'].append(train_metr)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['valid_metr'].append(valid_metr)
        history['unknown_valid_loss'].append(unknown_valid_loss)

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

def train(model, loader, optimizer, criterion, metric):
    model.train()

    train_loss = 0
    train_acc  = 0
    train_metr = 0
    with tqdm.tqdm(loader, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for idx, batch in enumerate(bar):
            metr = metric(model.embed(batch))

            true = np.repeat(np.arange(batch.shape[0]), batch.shape[1])
            data = torch.reshape(batch, (-1, batch.shape[2], batch.shape[3]))
            del batch

            optimizer.zero_grad()
            pred = model(data)
            del data

            true = torch.from_numpy(true).long().to('cuda')
            loss = criterion(pred, true)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += pred.argmax(dim=1).eq(true).sum().item()
            train_metr += metr.item()

            bar.set_postfix({
                'loss': '%.4f' % (train_loss / ((idx + 1) * len(true))),
                'acc': '%.2f %%' % (100 * train_acc / ((idx + 1) * len(true))),
                'ge2e': '%.4f' % (train_metr / ((idx + 1) * len(true)))
            })

    train_loss /= len(loader) * len(true)
    train_acc  /= len(loader) * len(true)
    train_metr /= len(loader) * len(true)
    return train_loss, train_acc, train_metr

def valid(model, loader, criterion, metric):
    model.eval()

    valid_loss = 0
    valid_acc  = 0
    valid_metr = 0
    for batch in loader:
        with torch.no_grad():
            metr = metric(model.embed(batch))

            true = np.repeat(np.arange(batch.shape[0]), batch.shape[1])
            data = torch.reshape(batch, (-1, batch.shape[2], batch.shape[3]))
            del batch
            
            pred = model(data)
            del data

            true = torch.from_numpy(true).long().to('cuda')
            loss = criterion(pred, true)

            valid_loss += loss.item()
            valid_acc  += pred.argmax(dim=1).eq(true).sum().item()
            valid_metr += metr.item()

    valid_loss /= len(loader) * len(true)
    valid_acc  /= len(loader) * len(true)
    valid_metr /= len(loader) * len(true)
    return valid_loss, valid_acc, valid_metr

def unknown_valid(model, loader, criterion):
    model.eval()

    batch_size = len(loader[0]) * len(loader[0][0])
    valid_loss = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
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
    args = get_args()

    # GPU番号
    if args['gpu'] is None:
        ngpus = torch.cuda.device_count()
        gpu_no = console_menu('使用するGPUを選択してください', [torch.cuda.get_device_name(i) for i in range(ngpus)] + ['使用しない'])
        if gpu_no < 0: exit(0)
        if gpu_no == ngpus:
            args['gpu'] = ''
        else:
            args['gpu'] = str(gpu_no)

    # 時間幅変形の方法
    if args['deform_type'] is None:
        deform_types = ['stretch', 'padding', 'variable']
        selected_no = console_menu('時間幅変形の方法を選択してください', deform_types)
        if selected_no < 0: exit(0)
        args['deform_type'] = deform_types[selected_no]

    # 出力先のパス
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    specific = {
        'deform_type': args['deform_type'],
        'code_id'    : args['code_id'],
    }
    output_dir_format = os.path.join(args['dest_dir'], '%(code_id)s/%(deform_type)s/%(datetime)s/')
    args['output_dir']      = output_dir_format % { **specific, 'datetime': now }
    args['output_wild_dir'] = output_dir_format % { **specific, 'datetime': '*' }

    # データセットのパス
    args['dataset_path'] = os.path.join(args['dataset_dir'], 'jvs%(person)03d/VOICEACTRESS100_%(voice)03d_%(deform_type)s.npz')

    cfg = easydict.EasyDict(args)

    main(cfg)
