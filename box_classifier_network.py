from torch import nn


class BoxClassifier(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv = pretrained

    def forward(self, _input):
        return _input
