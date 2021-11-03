import csv
import datetime
import logging
import os
import sys

import numpy as np
import sklearn.multiclass
import sklearn.svm
import torch
import tqdm

from emb_model import FullModel


class Utterances(object):

    def __init__(self, nsamples):
        self.nsamples = nsamples

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
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            uttrs = self.data[i]
            yield torch.from_numpy(np.array(uttrs)).float()

def main():
    code_id  = 'emb-main'
    run_id   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = os.path.join('./dest', code_id, run_id)
    os.makedirs(work_dir, exist_ok=True)
    init_logger(os.path.join(work_dir, 'general.log'))

    device = 'cuda:1'
    emb_weight_path = './dest/emb-train/20211103-182019/weights.pth'

    nsamples = 512

    model = FullModel(100).to(device).eval()
    model.load_state_dict(torch.load(emb_weight_path))
    model = model.embed

    dataset = Utterances(nsamples)

    est = []
    with tqdm.tqdm(dataset, bar_format='{l_bar}{bar:24}| [{elapsed}<{remaining}{postfix}]') as bar:
        for i, data in enumerate(bar):
            with torch.no_grad():
                est.append(model(data.to(device)).cpu().numpy())
            bar.set_description(f'{i}/{len(dataset)}')

    est = np.array(est)

    emb = np.mean(est, axis=1)
    np.save(os.path.join(work_dir, 'centroids.npy'), emb)

    confmat, ncorrects, ndata = calc_svm(est[:, :80], est[:, 80:])
    logging.info(f'Accuracy: {ncorrects / ndata} ({ncorrects}/{ndata})')
    with open(os.path.join(work_dir, 'svm_confmat.csv'), 'w') as f:
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

if __name__ == '__main__':
    main()
