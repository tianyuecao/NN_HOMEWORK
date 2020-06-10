import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.DANN import DANN
from dataset.SEED import SEEDDataLoader
from run.test import test_dann
import sys
import matplotlib.pyplot as plt


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


sys.stdout = Logger("log_dann.txt")

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# hyper parameters
lr = 0.001  # learning rate
max_epoch_num = 600
data_f = 'data.pkl'
batch_size = 8
in_dim = 310
fea_hid_dim = 128
cls_hid_dim = 64
class_num = 3
alpha = 1
model_root = './models'


def train(ti, s_dataloader, t_dataloader, DANN_Model):
    loss_class = nn.NLLLoss().to(device)
    loss_domain = nn.NLLLoss().to(device)
    optimizer = optim.SGD(DANN_Model.parameters(), lr=lr)

    plt.figure()
    epoches = []
    s_label_losses = []
    s_domain_losses = []
    t_domain_losses = []
    for epoch in range(max_epoch_num):

        DANN_Model.train()
        len_dataloader = min(len(s_dataloader), len(t_dataloader))
        source_iter = iter(s_dataloader)
        target_iter = iter(t_dataloader)

        for i in range(len_dataloader):
            # source data
            s_label_loss = 0.0
            s_domain_loss = 0.0
            for j in range(4):
                try:
                    data_source = source_iter.next()
                except:
                    break

                s_data, s_label = data_source
                s_data, s_label = s_data.to(device), s_label.to(device)
                s_domain = torch.zeros((len(s_data), 1), dtype=torch.long).to(device)

                DANN_Model.zero_grad()

                s_class_pred, s_domain_pred = DANN_Model(s_data)
                s_label = s_label.flatten()
                s_domain = s_domain.flatten()
                s_label_loss += loss_class(s_class_pred, s_label)
                s_domain_loss += loss_domain(s_domain_pred, s_domain)
            s_label_loss /= 4
            s_domain_loss /= 4

            # target data
            data_target = target_iter.next()
            t_data, _ = data_target
            t_data = t_data.to(device)
            t_domain = torch.ones((len(t_data), 1), dtype=torch.long).to(device)

            _, t_domain_pred = DANN_Model(t_data)
            t_domain = t_domain.flatten()
            t_domain_loss = loss_domain(t_domain_pred, t_domain)

            loss = s_label_loss + s_domain_loss + t_domain_loss
            loss.backward()
            optimizer.step()

        print('epoch: %d, s_label_loss: %f, s_domain_loss: %f, t_domain_loss: %f' \
              % (epoch, s_label_loss.cpu().data.numpy(),
                 s_domain_loss.cpu().data.numpy(), t_domain_loss.cpu().data.numpy()))

        epoches.append(epoch)
        s_label_losses.append(s_label_loss.cpu().data.numpy())
        s_domain_losses.append(s_domain_loss.cpu().data.numpy())
        t_domain_losses.append(t_domain_loss.cpu().data.numpy())

        torch.save(DANN_Model, '{0}/{1}_dann_model_epoch_{2}.pth'.format(model_root, ti, epoch))
        acc = test_dann(epoch, ti, t_dataloader)
    plt.plot(epoches, s_label_losses, c='orange')
    plt.plot(epoches, s_domain_losses, c='red')
    plt.plot(epoches, t_domain_losses, c='blue')
    plt.savefig('{}_dann_loss.jpg'.format(ti))


if __name__ == '__main__':
    # load data
    for i in range(5):
        s_list = [0, 1, 2, 3, 4]
        s_list.remove(i)
        t_list = [i]
        # load model
        DANN_Model = DANN(in_dim, fea_hid_dim, cls_hid_dim, class_num, alpha).to(device)
        source_dataset = SEEDDataLoader(data_f, data_split=s_list)
        source_dataloader = DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
        target_dataset = SEEDDataLoader(data_f, data_split=t_list)
        target_dataloader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        train(i, source_dataloader, target_dataloader, DANN_Model)

    print('done')
