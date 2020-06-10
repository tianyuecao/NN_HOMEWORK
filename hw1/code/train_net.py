# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import numpy as np
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import EmotionDataset
from model import SimpleNet

# draw fig
DRAW = 0

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# set random seed to fully reimplement the result
seed = 3
torch.manual_seed(seed)            # set random seed for cpu
torch.cuda.manual_seed_all(seed)       # set random seed for all gpu

# hyper parameters
lr = 0.001  # learning rate
max_epoch_num = 3000
root_dir = './train_test/'
hidden_layer = 256  # hidden layer dimension
batch_size = 4

# load data
train_data = EmotionDataset(root_dir='./train_test', dataset='train')
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
train_data_pytorch = torch.from_numpy(train_data.data).float().to(device)
train_label = train_data.labels.squeeze()

test_set = sio.loadmat(os.path.join(root_dir, 'test_data.mat'))['test_data']
test_label = (sio.loadmat(os.path.join(root_dir, 'test_label.mat'))['test_label'] + 1).squeeze()
test_data_pytorch = torch.from_numpy(test_set).float().to(device)
print("data load finished.")

# instantiate model
model = SimpleNet(310, hidden_layer, 3).to(device)


# training process
def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if DRAW:
        loss_data = []
        acc_data = []

    start = time.time()
    for ep in range(max_epoch_num):
        # train
        model.train()

        for i, data in enumerate(train_loader, 0):
            (emotions, labels) = data
            emotions, labels = emotions.to(device), labels.to(device)

            output = model(emotions)
            labels = labels.squeeze()
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # output statistics
        if (ep + 1) % 5 == 0:
            model.eval()
            p_out = model(train_data_pytorch)
            predict_labels = torch.argmax(p_out, 1).cpu().numpy()
            acc = np.sum(predict_labels == train_label) / len(predict_labels)
            print('epoch num : {} -- Loss : {} -- Acc : {}'.format(ep + 1, loss.data, acc))

        # draw fig
        if DRAW:
            loss_data.append(loss.data)
            model.eval()
            p_out = model(train_data_pytorch)
            predict_labels = torch.argmax(p_out, 1).cpu().numpy()
            acc = np.sum(predict_labels == train_label) / len(predict_labels)
            acc_data.append(acc)

    end = time.time()
    print('time :', end-start)

    if DRAW:
        plt.plot(loss_data, color='r')
        plt.plot(acc_data, color='b')
        plt.savefig('./outputs/fig'+str(hidden_layer)+
                    str(lr)+str(batch_size)+'.png')


# test process
def test():
    model.eval()
    test_output = model(test_data_pytorch)
    predict_labels = torch.argmax(test_output, 1).cpu().numpy()
    acc = np.sum(predict_labels == test_label) / len(test_label)
    print(acc)


if __name__ == '__main__':
    train()
    test()