import torch
import torch.nn.functional as F
from torch import nn


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#  AdaptiveAvgPool2d -> Conv2d -> 
class EmbeddingHead(nn.Module):
    """
    It typically contains logic to
    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """
    def __init__(self, feat_dim, embedding_dim, num_classes):
        super().__init__()
        # Pooling layer 都有
        self.gempool = GeneralizedMeanPooling()
        # self.se = SELayer(feat_dim)
        self.hswish = nn.Hardswish()
        self.pool_layer = nn.AdaptiveAvgPool2d(output_size=1)
        self.liner = nn.Linear(feat_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.cls_layer = nn.Linear(embedding_dim, num_classes)
        # self.liner.apply(weights_init_kaiming)
        # self.bn.apply(weights_init_kaiming)
        # self.cls_layer.apply(weights_init_kaiming)


    def forward(self, features):
        # print(features.shape)  torch.Size([16, 2048, 7, 7])
        # features = self.pool_layer(features)
        features = self.gempool(features)
        # features = self.se(features)
        features = self.hswish(features)
        # print(features.shape) torch.Size([16, 2048, 1, 1])
        features = torch.flatten(features, 1)
        features = self.liner(features)
        features = self.bn(features)

        if not self.training:
            return features
        cls_outputs = self.cls_layer(features)

        return {
            "cls_outputs": cls_outputs,
            "features": features,
        }
