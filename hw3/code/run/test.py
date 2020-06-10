import os
import torch.utils.data

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# hyper parameters
lr = 0.001  # learning rate
max_epoch_num = 3000
data_f = 'data.pkl'
hidden_layer = 256  # hidden layer dimension
batch_size = 8
in_dim = 310
fea_hid_dim = 128
cls_hid_dim = 64
class_num = 3
alpha = 1
model_root = './models/'
mode = "maxmin"


def test_baseline(epoch, ti, target_dataloader):
    model = torch.load(os.path.join(model_root, str(ti) + "_baseline_model_epoch_" + str(epoch) + '.pth')).to(device)
    model = model.eval()

    len_dataloader = len(target_dataloader)
    target_iter = iter(target_dataloader)

    total = 0
    correct = 0
    for i in range(len_dataloader):
        data_target = target_iter.next()
        t_data, t_label = data_target
        t_data, t_label = t_data.to(device), t_label.to(device)

        label_pred, _ = model(t_data)
        pred = label_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        total += batch_size

    acc = correct.data.numpy() * 1.0 / total

    print('epoch: %d, accuracy of the dataset: %f' % (epoch, acc))
    return acc


def test_dann(epoch, ti, target_dataloader):
    model = torch.load(os.path.join(model_root, str(ti) + "_dann_model_epoch_" + str(epoch) + '.pth')).to(device)
    model = model.eval()

    len_dataloader = len(target_dataloader)
    target_iter = iter(target_dataloader)

    total = 0
    correct = 0
    for i in range(len_dataloader):
        data_target = target_iter.next()
        t_data, t_label = data_target
        t_data, t_label = t_data.to(device), t_label.to(device)

        label_pred, _ = model(t_data)
        pred = label_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        total += batch_size

    acc = correct.data.numpy() * 1.0 / total

    print('epoch: %d, accuracy of the dataset: %f' % (epoch, acc))
    return acc


def test_mdan(epoch, ti, target_dataloader):
    model = torch.load(os.path.join(model_root, str(ti) + "_mdan_" + mode +
                                    "_model_epoch_" + str(epoch) + "_" + str(lr) + '.pth')).to(device)
    model = model.eval()

    len_dataloader = len(target_dataloader)
    target_iter = iter(target_dataloader)

    total = 0
    correct = 0
    for i in range(len_dataloader):
        data_target = target_iter.next()
        t_data, t_label = data_target
        t_data, t_label = t_data.to(device), t_label.to(device)

        label_pred, _ = model(t_data, 0)
        pred = label_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        total += batch_size

    acc = correct.data.numpy() * 1.0 / total

    print('epoch: %d, accuracy of the dataset: %f' % (epoch, acc))
    return acc
