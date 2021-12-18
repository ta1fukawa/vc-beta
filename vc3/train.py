import argparse
import datetime
import pathlib
import sys

import torch, torch.utils.tensorboard
import tqdm
import yaml

import model
import dataset

def main(mel_dir, embed_dir, dest_dir, config_path, model_path, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    run_id = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    dest_dir = dest_dir / run_id
    dest_dir.mkdir(exist_ok=True, parents=True)

    sw = torch.utils.tensorboard.SummaryWriter(dest_dir)

    if model_path is not None:
        net = torch.load(model_path).to(device)
    else:
        net = model.AutoVC(config['autovc']['config']).to(device)

    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))

    def creterion(src_mel, src_cnt, rec_mel, pst_mel, pst_cnt):
        weight = config['autovc']['weight']

        rec_loss = torch.nn.functional.mse_loss(rec_mel, src_mel)
        pst_loss = torch.nn.functional.mse_loss(pst_mel, src_mel)
        cnt_loss = torch.nn.functional.l1_loss(pst_cnt, src_cnt)
        loss = weight['rec'] * rec_loss + weight['pst'] * pst_loss + weight['cnt'] * cnt_loss
        
        return loss, (rec_loss, pst_loss, cnt_loss)

    def train(net, optimizer, train_loader, epoch, sw):
        net.train()

        with tqdm.tqdm(train_loader) as pbar:
            for step, (src_mel, src_emb) in enumerate(pbar):
                src_mel = src_mel.to(device)
                src_emb = src_emb.to(device)

                optimizer.zero_grad()
                src_cnt, rec_mel, pst_mel, pst_cnt = net(src_mel, src_emb)
                loss, loss_detail = creterion(src_mel, src_cnt, rec_mel, pst_mel, pst_cnt)
                loss.backward()
                optimizer.step()

                sw.add_scalar('loss', loss.item(), step + epoch * len(train_loader))
                sw.add_scalar('rec_loss', loss_detail[0].item(), step + epoch * len(train_loader))
                sw.add_scalar('pst_loss', loss_detail[1].item(), step + epoch * len(train_loader))
                sw.add_scalar('cnt_loss', loss_detail[2].item(), step + epoch * len(train_loader))
                
                pbar.set_description(f'Epoch {epoch}')
                pbar.set_postfix(loss=loss.item())
    
    train_loader = torch.utils.data.DataLoader(
        dataset.MelEmbLoader(mel_dir, embed_dir, **config['data']),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=False,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=config['train']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['lr_step'], gamma=config['train']['lr_gamma'])

    for epoch in range(config['train']['epochs']):
        train(net, optimizer, train_loader, epoch, sw)
        scheduler.step()

    torch.save(net.state_dict(), dest_dir / 'model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wav to mel spectrogram')
    parser.add_argument('mel_dir',       type=pathlib.Path, help='directory of mel spectrograms')
    parser.add_argument('embed_dir',     type=pathlib.Path, help='directory of embeddings')
    parser.add_argument('dest_dir',      type=pathlib.Path, help='destination directory')
    parser.add_argument('config_path',   type=pathlib.Path, help='path to config')
    parser.add_argument('--model_path',  type=pathlib.Path, help='path to network model')
    parser.add_argument('--weight_path', type=pathlib.Path, help='path to network weight')

    if 'debugpy' in sys.modules:
        args = parser.parse_args([
            'vc3/mel-jvs',
            'vc3/embed-jvs',
            'vc3/dest',
            'vc3/train.yaml',
        ])
    else:
        args = parser.parse_args([])

    main(**vars(args))
