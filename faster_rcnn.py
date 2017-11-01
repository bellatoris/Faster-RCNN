import torch
from torch import nn


class FasterRCNN(nn.Module):
    def __init__(self, pretrained_clf):
        super().__init__()
        self.features = nn.Sequential(
            *list(pretrained_clf.children())[:-1]
        )
        self.feature_extractor = FeatureExtractor()

    def forward(self, _input):
        return _input


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input):
        return _input


class ProposalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, bias=False,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, (4 + 2) * 9, kernel_size=1,
                               bias=False)

    def forward(self, _input):
        out = self.conv1(_input)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class BoxClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input):
        return _input
