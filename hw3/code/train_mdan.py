import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.MDAN import MDAN
from dataset.SEED import SEEDDataLoader
from run.test import test_mdan
import sys
import math


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

# set random seed to fully reimplement the result
seed = 3
torch.manual_seed(seed)  # set random seed for cpu
torch.cuda.manual_seed_all(seed)  # set random seed for all gpu

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# hyper parameters
lr = 0.003  # learning rate
max_epoch_num = 600
data_f = 'data.pkl'
hidden_layer = 256  # hidden layer dimension
batch_size = 8
in_dim = 310
fea_hid_dim = 128
cls_hid_dim = 64
class_num = 3
alpha = 1
model_root = './models'
num_data_sets = 5
num_domains = 4
mode = "maxmin"
gamma = 10.0

sys.stdout = Logger("log_" + mode + "_mdan_" + str(lr) + ".txt")

# load data
dataset_0 = SEEDDataLoader(data_f, data_split=[0])
dataloader_0 = DataLoader(dataset=dataset_0, batch_size=batch_size, shuffle=True)
dataset_1 = SEEDDataLoader(data_f, data_split=[1])
dataloader_1 = DataLoader(dataset=dataset_1, batch_size=batch_size, shuffle=True)
dataset_2 = SEEDDataLoader(data_f, data_split=[2])
dataloader_2 = DataLoader(dataset=dataset_2, batch_size=batch_size, shuffle=True)
dataset_3 = SEEDDataLoader(data_f, data_split=[3])
dataloader_3 = DataLoader(dataset=dataset_3, batch_size=batch_size, shuffle=True)
dataset_4 = SEEDDataLoader(data_f, data_split=[4])
dataloader_4 = DataLoader(dataset=dataset_4, batch_size=batch_size, shuffle=True)
dataloaders = [dataloader_0, dataloader_1, dataloader_2, dataloader_3, dataloader_4]


def train(ti, MDAN_Model):
    loss_class = nn.NLLLoss().to(device)
    loss_domain = nn.NLLLoss().to(device)
    optimizer = optim.SGD(MDAN_Model.parameters(), lr=lr)

    for epoch in range(max_epoch_num):
        MDAN_Model.train()
        len_dataloader = min(len(dataloader_0), len(dataloader_1),
                             len(dataloader_2), len(dataloader_3), len(dataloader_4))
        iters = [iter(dataloader_0), iter(dataloader_1),
                 iter(dataloader_2), iter(dataloader_3), iter(dataloader_4)]
        s_list = [0, 1, 2, 3, 4]
        s_list.remove(ti)

        for _ in range(len_dataloader):
            # target data
            data_target = iters[ti].next()
            t_data, _ = data_target
            t_data = t_data.to(device)
            t_domain = torch.ones((len(t_data), 1), dtype=torch.long).to(device)

            # source data
            label_losses = []
            domain_losses = []
            i = 0
            for si in s_list:
                data_source = iters[si].next()
                s_data, s_label = data_source
                s_data, s_label = s_data.to(device), s_label.to(device)
                s_domain = torch.zeros((len(s_data), 1), dtype=torch.long).to(device)

                MDAN_Model.zero_grad()

                s_class_pred, s_domain_pred = MDAN_Model(s_data, i)
                s_label = s_label.flatten()
                s_domain = s_domain.flatten()
                _, t_domain_pred = MDAN_Model(t_data, i)
                t_domain = t_domain.flatten()

                label_losses.append(loss_class(s_class_pred, s_label))
                domain_losses.append(loss_domain(s_domain_pred, s_domain) +
                                     loss_domain(t_domain_pred, t_domain))
                i += 1

            if mode == "maxmin":
                loss = max(label_losses) + min(domain_losses)
            elif mode == "dynamic":
                loss = torch.log(sum(torch.exp(label_losses[k]) + torch.exp(domain_losses[k]) for k in range(4)))
            else:
                raise ValueError("No support for the training mode on madnNet: {}.".format(mode))

            loss.backward()
            optimizer.step()

        if mode == "maxmin":
            print('epoch: %d, label_loss: %f, domain_loss: %f' \
                  % (epoch, max(label_losses).cpu().data.numpy(),
                     min(domain_losses).cpu().data.numpy()))
        elif mode == "dynamic":
            print('epoch: %d, label_loss: %f, domain_loss: %f' \
                  % (epoch, sum(label_losses).cpu().data.numpy(),
                     sum(domain_losses).cpu().data.numpy()))

        torch.save(MDAN_Model, '{0}/{1}_mdan_{2}_model_epoch_{3}_{4}.pth'
                   .format(model_root, ti, mode, epoch, lr))
        acc = test_mdan(epoch, ti, dataloaders[ti])


if __name__ == '__main__':
    for i in range(num_data_sets):
        # load model
        MDAN_Model = MDAN(in_dim, num_domains, fea_hid_dim, cls_hid_dim, class_num, alpha, device).to(device)
        train(i, MDAN_Model)

    print('done')
