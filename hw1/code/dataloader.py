import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class EmotionDataset(Dataset):
    """Emotion Dataset"""

    def __init__(self, root_dir, dataset="train", transform=None):
        """
        :param root_dir: dataset root directory
        :param type: "train" or "test"
        :param transform: data transformation
        """
        super(EmotionDataset).__init__()
        self.root_dir = root_dir
        self.transform = transform
        assert dataset == "train" or dataset == "test", \
            "dataset type can only be \'train\' or \'test\'."
        self.data = sio.loadmat(os.path.join(root_dir, dataset + '_data.mat'))[dataset + '_data']
        self.labels = sio.loadmat(os.path.join(root_dir, dataset + '_label.mat'))[dataset + '_label'] + 1
        assert len(self.data) == len(self.labels), \
            "data size should be the same as label size."

    def __getitem__(self, idx):
        emotion = self.data[idx].astype(np.float32)
        emotion = torch.from_numpy(emotion)
        label = self.labels[idx].astype(np.int64)
        label = torch.from_numpy(label)
        item = (emotion, label)

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    emotion_dataset = EmotionDataset(root_dir='./train_test',
                                     dataset='train')

    for i in range(len(emotion_dataset)):
        sample = emotion_dataset[i]

        print(i, type(sample['emotion']), sample['emotion'].dtype,
              type(sample['label']), sample['label'].dtype)
        if i == 4:
            break
