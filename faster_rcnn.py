import numpy as np
import torch
from torch import nn

from box_classifier_network import BoxClassifier
from build_detection_target import build_detection_target
from build_rpn_target import build_rpn_targets
from generate_anchors import generate_anchors
from region_proposal_network import ProposalGenerator, Proposal
from utils import get_variable_from_numpy


class FasterRCNN(nn.Module):
    def __init__(self, pretrained_clf, criterion, config):
        super().__init__()
        children = list(pretrained_clf.children())
        self.feature_extractor = nn.Sequential(*children[:7])
        self.proposal_generator = ProposalGenerator()
        self.proposal = Proposal()
        self.box_classifier = BoxClassifier(children[7])
        self.criterion = criterion
        self.config = config
        # self.box_classifier = BoxClassifier()

        # Freeze batch normalization weights
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, _input, gt_boxes):
        # get config
        scales = self.config['scales']
        ratios = self.config['ratios']
        feature_stride = self.config['feature_stride']
        anchor_stride = self.config['anchor_stride']
        rpn_batch_size = self.config['rpn_batch_size']
        num_proposals = self.config['num_proposals']

        # extract feature map
        feature = self.feature_extractor(_input)

        # get match class and bounding box
        match, bbox = self.proposal_generator(feature)

        # get argument for rpn loss
        feature_shape = feature.shape[2:]
        anchors = generate_anchors(scales, ratios, feature_shape,
                                   feature_stride, anchor_stride)

        inds_inside = get_inds_inside(_input.shape[2:], anchors)
        anchors_inside = anchors[inds_inside]

        # make rpn target
        target_match, target_bbox = build_rpn_targets(anchors_inside,
                                                      gt_boxes,
                                                      rpn_batch_size)

        # get match and bbox only inside the image
        inds_inside = get_variable_from_numpy(inds_inside)
        match_inside = match[inds_inside]
        bbox_inside = bbox[inds_inside]

        match_loss, bbox_loss = make_rpn_loss(target_match, target_bbox,
                                              match_inside, bbox_inside,
                                              self.criterion)

        match_loss = match_loss / rpn_batch_size
        bbox_loss = bbox_loss / (feature_shape[0] * feature_shape[1])

        proposal_bbox = self.proposal(match_inside, bbox_inside,
                                      anchors_inside, _input.shape[2:],
                                      num_proposals)

        rois, class_ids, deltas = build_detection_target(proposal_bbox.data,
                                                         gt_boxes)

        return match_loss, bbox_loss


def get_inds_inside(img_shape, anchors):
    # Find anchors inside image
    inds_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] < img_shape[0]) &  # height
        (anchors[:, 3] < img_shape[1])  # width
    )[0]

    return inds_inside


def make_rpn_loss(target_match, target_bbox, match, bbox, criterion):
    match_inds = np.where(target_match != 0)[0]
    positive_inds = np.where(target_match == 1)[0]

    target_match = target_match[match_inds]
    target_match += 1
    target_match = np.divide(target_match, 2)

    target_match = get_variable_from_numpy(target_match.astype(int))
    target_bbox = get_variable_from_numpy(target_bbox.astype(np.float32))
    match_inds = get_variable_from_numpy(match_inds)
    positive_inds = get_variable_from_numpy(positive_inds)

    match_loss = criterion['cross_entropy'](match[match_inds], target_match)
    bbox_loss = criterion['smooth_l1'](bbox[positive_inds],
                                       target_bbox[:len(positive_inds)])

    return match_loss, bbox_loss
