#!/usr/bin/env python
# -*-coding:utf-8-*-

import torch
import scipy.io
import os
from dataloader.dataloader import train_data_generator
from Configs import Config as Configs
from models.CP import CP
from models.model import base_Model
from models.loss import NTXentLoss
import torch.nn.functional as F
import argparse


def train_whole(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, device, train_loader, config):
    total_loss = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (_, __, aug1, aug2) in enumerate(train_loader):
        # send to device
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        _, features1 = model(aug1)
        _, features2 = model(aug2)

        # normalize projection feature vectors
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
        temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

        lambda1 = 1
        lambda2 = 0.5
        nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                       config.Context_Cont.use_cosine_similarity)
        loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(temp_cont_lstm_feat1, temp_cont_lstm_feat2) * lambda2

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    return total_loss

def load_dataset(mat_path, train_num):
    """
    Load and reshape dataset from .mat file.
    """
    filemat = scipy.io.loadmat(mat_path)
    signals = torch.tensor(filemat['merge_signal'])[1:]
    noises = torch.tensor(filemat['merge_noise'])[1:]

    random_indices = torch.randperm(10000)[:train_num]
    signal_data = signals[:, random_indices, :, :]
    noise_data = noises[:, random_indices, :, :]

    signals = torch.reshape(signal_data, (10 * train_num, 2, 1024))
    noises = torch.reshape(noise_data, (10 * train_num, 2, 1024))

    signal_labels = torch.ones(signals.shape[0], dtype=torch.long)
    noise_labels = torch.zeros(noises.shape[0], dtype=torch.long)

    samples = torch.cat((signals, noises), dim=0)
    labels = torch.cat((signal_labels, noise_labels), dim=0)

    return {'samples': samples, 'labels': labels}

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-Train of DC4S")

    parser.add_argument('--train_num', type=int, default=10000,
                        help='Number of training samples (per class before reshaping)')
    parser.add_argument('--mat_path', type=str, default=f'/home/dataset/OFDM_Dataset_Pretrain_-20to0_TDLe_len1024.mat',
                        help='Path to the input .mat file')
    parser.add_argument('--load_pretrained', default=False,
                        help='Whether to load pretrained weights')
    parser.add_argument('--save_lastmodel', default=False,
                        help='Whether to save the last model checkpoint')
    parser.add_argument('--save_ckpt_path', type=str, default='ckpt/scale_reverse.pt',
                        help='Path to save the final model checkpoint')

    return parser.parse_args()

def main():
    args = parse_args()

    # === Load Dataset ===
    train_dataset = load_dataset(args.mat_path, args.train_num)
    configs = Configs()
    train_dl = train_data_generator(train_dataset, configs)

    # === Set Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Initialize Models ===
    model = base_Model(configs).to(device)
    temporal_contr_model = CP(configs, device).to(device)

    # === Optimizers ===
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    # === Training ===
    for epoch in range(1, configs.num_epoch + 1):
        total_loss = train_whole(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, device, train_dl, configs)
        print(f'\nEpoch : {epoch}\n    Train Loss : {total_loss:.4f}')

    # === Save Final Model ===
    if args.save_lastmodel:
        os.makedirs(os.path.dirname(args.save_ckpt_path), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
        torch.save(chkpoint, args.save_ckpt_path)

if __name__ == '__main__':
    main()