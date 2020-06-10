import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

DEBUG = 0
DRAW = 0

# hyper parameters
KERNEL = 'poly'
C = 1
Cs = [1e-8, 1e-6, 1e-4, 1]
kernels = ['linear', 'rbf', 'poly']
GAMMA = 0.1
DEGREE = 3
num_class = 3
instance_num = 11
split = 2

# load dataset
root_dir = "./data_hw2"
minmax = MinMaxScaler()
train_data = np.load(os.path.join(root_dir, 'train_data.npy'))
train_data = minmax.fit_transform(train_data)
train_label = np.load(os.path.join(root_dir, 'train_label.npy')) + 1
test_data = np.load(os.path.join(root_dir, 'test_data.npy'))
test_data = minmax.fit_transform(test_data)
test_label = np.load(os.path.join(root_dir, 'test_label.npy')) + 1
train_num = len(train_data)
test_num = len(test_data)
print("Data load finished.")

if DEBUG:
    # data of different classes
    ind0 = np.where(train_label == 0)
    train_data0, train_label0, num0 = train_data[ind0], train_label[ind0], len(ind0[0])
    ind1 = np.where(train_label == 1)
    train_data1, train_label1, num1 = train_data[ind1], train_label[ind1], len(ind1[0])
    ind2 = np.where(train_label == 2)
    train_data2, train_label2, num2 = train_data[ind2], train_label[ind2], len(ind2[0])
    print('Number of instances in class -1 :', num0)
    print('Number of instances in class 0 :', num1)
    print('Number of instances in class 1 :', num2)

# prior
gender = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)

# split label
train_labels = []
test_labels = []
for i in range(num_class):
    train_label_ = np.where(train_label == i, 1, 0)
    train_labels.append(train_label_)
    test_label_ = np.where(test_label == i, 1, 0)
    test_labels.append(test_label_)

# split each label to 2 parts by gender
# split_ind = [[gender_0_num],[gender_1_num]]
split_tmp = np.arange(train_num)
split_tmp = split_tmp.reshape((instance_num, -1))
gender1 = split_tmp[np.where(gender == 1)].flatten()
gender0 = split_tmp[np.where(gender == 0)].flatten()
split_ind = [gender0, gender1]


def draw_ROC(test_preds, kernel, c):
    # draw ROC curve and calculate AUC
    plt.figure()
    lw = 2
    colors = ['#FE3C41', '#FF9F3C', '#38B2D5']
    for i in range(num_class):
        fpr, tpr, thresh = roc_curve(test_labels[i], test_preds[i], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i],
                 lw=lw, label='class(' + str(i - 1) + ') ROC curve (AUC = %0.2f)' % roc_auc)

    test_label_all = np.concatenate((test_labels[0], test_labels[1], test_labels[2]))
    test_pred_all = np.concatenate((test_preds[0], test_preds[1], test_preds[2]))
    fpr_all, tpr_all, thresholds_all = roc_curve(test_label_all, test_pred_all, pos_label=1)
    roc_auc_all = auc(fpr_all, tpr_all)
    plt.plot(fpr_all, tpr_all, color='darkgray', linestyle='-.',
             lw=6, label='average ROC curve (AUC = %0.2f)' % roc_auc_all)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of SVM')
    plt.legend(loc="lower right")
    plt.savefig('outputs/problem2_prior/roc' + kernel + str(c) + '.png')
    # plt.show()


def train_net(train, test, kernel, c):
    # classifier
    clfs = []
    if train:
        for i in range(num_class):
            for s in range(split):
                clf = CalibratedClassifierCV(svm.SVC(kernel=kernel, C=c,
                                                     probability=True), cv=5)
                clf.fit(train_data[split_ind[s], :], train_labels[i][split_ind[s]])
                clfs.append(clf)
                print("classifier", i, "for split", s, "train finished.")

        with open('./outputs/problem2_prior/' + kernel + str(c) + 'model.pkl', 'wb') as f:
            pkl.dump(clfs, f)
        print('Model dumped.')

    if test:
        with open('./outputs/problem2_prior/' + kernel + str(c) + 'model.pkl', 'rb') as f:
            clfs = pkl.load(f)
        print('Model load finished.')
        test_pred = []
        train_pred = []
        test_preds = np.zeros((num_class, split, test_num))
        test_scores = np.zeros((num_class, test_num))
        for i in range(num_class):
            for s in range(split):
                test_preds[i][s] = clfs[i * split + s].predict_proba(test_data)[:, 1]
                print('Prediction class', i, 'split', s, 'finished.')

        # MINMAX Module
        # MIN
        for i in range(num_class):
            test_scores[i] = np.min(test_preds[i], axis=0)

        # MAX
        test_score = np.max(test_scores, axis=0)
        test_output = np.argmax(test_scores, axis=0)
        print(test_output)
        print(metrics.accuracy_score(test_label, test_output))
        print(metrics.classification_report(test_label, test_output))

        if DRAW:
            draw_ROC(test_scores, kernel, c)


if __name__ == '__main__':
    train_net(1, 1, KERNEL, C)
