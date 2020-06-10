import torch
import torch.nn as nn
from model.newlayer import GradientReversalLayer


class MDAN(nn.Module):
    """MDAN model."""

    def __init__(self, in_dim, num_domains, fea_hid_dim, cls_hid_dim, class_num, alpha, ctx):
        """
        :param in_dim: input dimension
        :param num_domains: number of source domains
        :param fea_hid_dim: feature extractor hidden layer dimension
        :param cls_hid_dim: label classifier hidden layer dimension
        :param class_num: class number
        :param alpha: balanced hyperparameter
        :param ctx: divice
        """
        super(MDAN, self).__init__()
        self.in_dim = in_dim
        self.num_domains = num_domains
        self.fea_hid_dim = fea_hid_dim
        self.cls_hid_dim = cls_hid_dim
        self.class_num = class_num
        self.alpha = alpha
        self.ctx = ctx

        # feature extractors
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

        # domain classifiers
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fea_hid_dim, cls_hid_dim),
                nn.Tanh(),
                nn.Linear(cls_hid_dim, cls_hid_dim),
                nn.Tanh(),
                nn.Linear(cls_hid_dim, 2),
                nn.LogSoftmax()
            ) for _ in range(self.num_domains)])

    def forward(self, x, i):
        # feature extraction
        feature = self.feature(x)
        reverse_feature = GradientReversalLayer.apply(feature, self.alpha)
        # label classification on source domains
        label_pred = self.label_classifier(feature)
        # domain classification
        domain_pred = self.domain_classifiers[i](reverse_feature)
        return label_pred, domain_pred
