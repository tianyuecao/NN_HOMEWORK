import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SEEDDataLoader(Dataset):
    """SEED Dataset"""

    def __init__(self, data_f, data_split: list, transform=None):
        """
        :param data_f: dataset file
        :param data_split: a list contains 0, 1, 2, 3, or 5
        :param transform: data transformation
        """
        super(SEEDDataLoader).__init__()
        self.data_f = data_f
        self.transform = transform
        minmax = MinMaxScaler()  # data normalization operator

        assert set(data_split).issubset({0, 1, 2, 3, 4}), \
            "data_split can only be a subset of {0, 1, 2, 3, 4}."
        with open(data_f, 'rb') as f:
            data = pickle.load(f)

        self.data = np.zeros((len(data_split) * 3397, 310), dtype=np.float32)
        for i in range(len(data_split)):
            item = 'sub_' + str(data_split[i])
            self.data[i * 3397:(i + 1) * 3397, :] = np.stack(minmax.fit_transform(data[item]['data']), axis=0)

        self.label = np.zeros((len(data_split) * 3397, 1), dtype=np.int64)
        for i in range(len(data_split)):
            item = 'sub_' + str(data_split[i])
            self.label[i * 3397:(i + 1) * 3397, :] = \
                np.reshape(np.stack(data[item]['label'] + 1, axis=0), (-1, 1))
        assert len(self.data) == len(self.label), \
            "data size should be the same as label size."

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.from_numpy(data)
        label = self.label[idx]
        label = torch.from_numpy(label)
        item = (data, label)

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    seed = SEEDDataLoader(data_f='../data.pkl', data_split=[0, 1, 2, 3])

    for i in range(len(seed)):
        sample = seed[i]

        print(i, type(sample[0]), sample[0].dtype,
              type(sample[1]), sample[1].dtype)
        if i == 4:
            break

