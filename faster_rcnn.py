from torch import nn


class FasterRCNN(nn.Module):
    def __init__(self, pretrained_clf):
        super().__init__()
        children = list(pretrained_clf.children())
        self.feature_extractor = nn.Sequential(*children[:7])
        self.proposal_generator = ProposalGenerator()
        # self.box_classifier = BoxClassifier()

        # Freeze batch normalization weights
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, _input):
        feature = self.feature_extractor(_input)
        cls, bbox = self.proposal_generator(feature)
        return feature, cls, bbox


class ProposalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, bias=False,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, (4 + 2) * 9, kernel_size=1,
                               bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, _input):
        out = self.conv1(_input)
        out = self.relu(out)
        out = self.conv2(out)
        # out's shape is [1, 54, H, W] and H * W is the number of anchors
        out = out.view(out.size(1), out.size(2), out.size(3))
        out = out.transpose(0, 1)
        out = out.transpose(1, 2)  # [H, W, 54]
        out = out.contiguous()
        out = out.view(out.size(0), out.size(1), 9, 6)
        out = out.view(-1, 6)

        cls = out[:, :2]  # [9xHxW, 2]
        bbox = out[:, 2:]  # [9xHxW, 4]

        return cls, bbox


# class BoxClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, *_input):
#         return _input
