import numpy as np
import torch
from torch import nn

from utils import apply_box_deltas, get_variable_from_numpy


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

        match = out[:, :2]  # [9xHxW, 2]
        bbox = out[:, 2:]  # [9xHxW, 4]

        return match, bbox


class Proposal(nn.Module):
    """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinment detals to anchors.

        Inputs:
            rpn_probs: [anchors, (bg prob, fg prob)]
            rpn_bbox: [anchors, (dy, dx, log(dh), log(dw))]

        Returns:
            Proposals in normalized coordinates [rois, (y1, x1, y2, x2)]
        """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, match, bbox, anchors, img_shape, num_proposals):
        scores = self.softmax(match)
        height, width = img_shape

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        bboxes = apply_box_deltas(get_variable_from_numpy(anchors.astype(np.float32)),
                                  bbox)

        # Normalize dimensions to range of 0 to 1.
        # normalized_bboxes = bboxes / torch.from_numpy(np.array([[height,
        #                                                          width,
        #                                                          height,
        #                                                          width]]))
        proposals_ids = self.nms(bboxes.data,
                                 scores.data[:, 1],
                                 0.6,
                                 num_proposals)
        proposals = bboxes[proposals_ids]

        return proposals

    def nms(self, bboxes, scores, threshold, num_proposals, mode='union'):
        """Non maximum suppression.
        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          num_proposals: (int) max number of proposals
          mode: (str) 'union' or 'min'.
        Returns:
          keep: (tensor) selected indices.
        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1 or len(keep) == num_proposals:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        return torch.cuda.LongTensor(keep)
