import argparse
import csv
import datetime
import logging
import os
import shutil
import sys

import numpy as np
import sklearn.multiclass
import sklearn.svm
import torch
import tqdm

from emb_model import FullModel


def main():
    CODE_ID  = 'emb-main'
    RUN_ID   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    WORK_DIR = os.path.join('./dest', CODE_ID, RUN_ID)
    LOG_PATH = os.path.join(WORK_DIR, 'log.txt')

    os.makedirs(WORK_DIR, exist_ok=True)
    init_logger(LOG_PATH)
    logging.info(f'Output: {WORK_DIR}')
    backup_codes(['./autovc-fork/emb_main.py', './autovc-fork/emb_model.py'], WORK_DIR)

    args = get_args()
    logging.info(args)

    model = FullModel(100).to(args.device).eval()
    model.load_state_dict(torch.load(args.emb_weight_path))
    model = model.embed

    dataset = Utterances(args.nsmpls, args.sp_path)

    est = []
    with tqdm.tqdm(dataset, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for i, data in enumerate(bar):
            with torch.no_grad():
                est.append(model(data.to(args.device)).cpu().numpy())
            bar.set_description(f'{i}/{len(dataset)}')

    est = np.array(est)

    emb = np.mean(est, axis=1)
    np.save(os.path.join(WORK_DIR, 'centroids.npy'), emb)

    confmat, ncorrects, ndata = calc_svm(est[:, :80], est[:, 80:])
    logging.info(f'Accuracy: {ncorrects / ndata} ({ncorrects}/{ndata})')
    with open(os.path.join(WORK_DIR, 'svm_confmat.csv'), 'w') as f:
        csv.writer(f).writerows(confmat)

    train_nspkrs = int(len(est) * args.train_spkr_rate)

    confmat, ncorrects, ndata = calc_svm(est[:train_nspkrs, :80], est[:train_nspkrs, 80:])
    logging.info(f'Train accuracy: {ncorrects / ndata} ({ncorrects}/{ndata})')
    with open(os.path.join(WORK_DIR, 'svm_confmat_train.csv'), 'w') as f:
        csv.writer(f).writerows(confmat)

    confmat, ncorrects, ndata = calc_svm(est[train_nspkrs:, :80], est[train_nspkrs:, 80:])
    logging.info(f'Eval. accuracy: {ncorrects / ndata} ({ncorrects}/{ndata})')
    with open(os.path.join(WORK_DIR, 'svm_confmat_eval.csv'), 'w') as f:
        csv.writer(f).writerows(confmat)

def init_logger(log_path, mode='w', stdout=True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_path, filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sp_path', type=str, default='./resource/sp/phonemes_v1')
    parser.add_argument('--emb_weight_path', type=str, default='./dest/emb-train/20211103-182019/weights.pth')
    parser.add_argument('--device',  type=str, default='cuda:0')

    parser.add_argument('--nsmpls', type=int, default=32)

    parser.add_argument('--train_spkr_rate', type=float, default=0.8)

    args = parser.parse_args()
    return args

def backup_codes(src_files, dest_dir):
    for src_file in src_files:
        shutil.copyfile(src_file, os.path.join(dest_dir, os.path.split(src_file)[1]))

def calc_confmat(pred, true, nclasses):
    confmat = np.zeros((nclasses, nclasses))
    for eval_true_label, pred_label in zip(true, pred):
        confmat[eval_true_label][pred_label] += 1
    return confmat

def calc_svm(train_data, eval_data):
    n_classes, n_train_items, embed_dim = train_data.shape
    n_classes, n_eval_items,  embed_dim = eval_data.shape

    train_label = np.concatenate([[person_idx] * n_train_items for person_idx in range(n_classes)])
    svm = sklearn.svm.SVC(C=1., kernel='rbf')
    classifier = sklearn.multiclass.OneVsRestClassifier(svm)
    classifier.fit(train_data.reshape(-1, embed_dim), train_label)

    eval_pred  = classifier.predict(eval_data.reshape(-1, embed_dim))
    eval_label = np.concatenate([[person_idx] * n_eval_items for person_idx in range(n_classes)])
    confmat = calc_confmat(eval_pred, eval_label, n_classes)

    n_corrects = np.trace(confmat).astype(np.int)
    n_data = (n_classes * n_eval_items)
    return confmat, n_corrects, n_data

class Utterances(object):

    def __init__(self, nsamples, path):
        self.nsamples = nsamples

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

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            uttrs = self.data[i]
            yield torch.from_numpy(np.array(uttrs)).float()

if __name__ == '__main__':
    main()
