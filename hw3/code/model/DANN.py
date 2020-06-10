import torch.nn as nn
from model.newlayer import GradientReversalLayer


class DANN(nn.Module):
    """DANN model."""

    def __init__(self, in_dim, fea_hid_dim, cls_hid_dim, class_num, alpha):
        """
        :param in_dim: input dimension
        :param fea_hid_dim: feature extractor hidden layer dimension
        :param cls_hid_dim: label classifier hidden layer dimension
        :param class_num: class number
        :param alpha: balanced parameter
        """
        super(DANN, self).__init__()
        self.in_dim = in_dim
        self.fea_hid_dim = fea_hid_dim
        self.cls_hid_dim = cls_hid_dim
        self.class_num = class_num
        self.domain_num = 2
        self.alpha = alpha

        # feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, fea_hid_dim),
            nn.Tanh(),
            nn.Linear(fea_hid_dim, fea_hid_dim),
            nn.Sigmoid()
        )

        # label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(fea_hid_dim, cls_hid_dim),
            nn.Tanh(),
            nn.Linear(cls_hid_dim, cls_hid_dim),
            nn.Tanh(),
            nn.Linear(cls_hid_dim, class_num),
            nn.LogSoftmax()
        )

        # domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(fea_hid_dim, cls_hid_dim),
            nn.Tanh(),
            nn.Linear(cls_hid_dim, cls_hid_dim),
            nn.Tanh(),
            nn.Linear(cls_hid_dim, self.domain_num),
            nn.LogSoftmax()
        )

    def forward(self, x):
        feature = self.feature(x)
        reverse_feature = GradientReversalLayer.apply(feature, self.alpha)
        label_pred = self.label_classifier(feature)
        domain_pred = self.domain_classifier(reverse_feature)
        return label_pred, domain_pred
