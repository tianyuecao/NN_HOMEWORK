from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pickle


# load dataset
minmax = MinMaxScaler()
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
print("Data load finished.")


def draw_graph():
    # 绘制主成分增加对原数据的保留信息影响
    # pca = PCA(n_components=X_train.shape[1])
    # pca.fit(X_train)
    # plt.plot([i for i in range(X_train.shape[1])],
    #          [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(X_train.shape[1])])
    # plt.show()
    # 把64维降维2维，进行数据可视化
    features = []
    for item in list(data.keys()):
        features.append(data[item]['data'])
    features = np.array(features).reshape(-1, 310)

    pca = PCA(n_components=2)
    pca.fit(features)
    X_reduction = pca.transform(features)
    X_reduction_group = X_reduction.reshape((5, -1, 2))

    for i in range(5):
        plt.scatter(X_reduction_group[i, :, 0], X_reduction_group[i, :, 1],
                    alpha=0.8, linewidths=0, label='%s' % i)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_graph()
