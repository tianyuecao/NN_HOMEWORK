from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from timeit import timeit
import numpy as np
import os

# load dataset
root_dir = "./data_hw2"
minmax = MinMaxScaler()
train_data = np.load(os.path.join(root_dir, 'train_data.npy'))
train_data = minmax.fit_transform(train_data)
train_label = np.load(os.path.join(root_dir, 'train_label.npy')) + 1
test_data = np.load(os.path.join(root_dir, 'test_data.npy'))
test_data = minmax.fit_transform(test_data)
test_label = np.load(os.path.join(root_dir, 'test_label.npy')) + 1
print("Data load finished.")


def draw_graph():
    # 绘制主成分增加对原数据的保留信息影响
    # pca = PCA(n_components=X_train.shape[1])
    # pca.fit(X_train)
    # plt.plot([i for i in range(X_train.shape[1])],
    #          [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(X_train.shape[1])])
    # plt.show()
    # 把64维降维2维，进行数据可视化
    pca = PCA(n_components=2)
    pca.fit(train_data)
    X_reduction = pca.transform(train_data)
    X_reduction_group = X_reduction.reshape((11, -1, 2))
    gender = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    male = X_reduction_group[np.where(gender == 1)]
    female = X_reduction_group[np.where(gender == 0)]

    colors = {1: "blue", 0: "red"}

    # draw by gender
    plt.scatter(male[:, :, 0], male[:, :, 1], label='1',
                    c=colors[1], linewidths=0, alpha=0.8)
    plt.scatter(female[:, :, 0], female[:, :, 1], label='0',
                c=colors[0], linewidths=0, alpha=0.8)

    '''
    # draw by people
    for i in range(11):
        plt.scatter(X_reduction_group[i, :, 0], X_reduction_group[i, :, 1],
                    alpha=0.8, linewidths=0, label='%s' % i)
    '''

    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_graph()
