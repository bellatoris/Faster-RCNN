import torch
import torch.nn.functional as F

from utils import box_refinement


def build_detection_target(proposals, gt_boxes):
    """Subsamples proposals and generates target box refinement, class_ids,
     and masks for each.

     Inputs:
     proposals: [N, (y1, x1, y2, x2)] Might be zero padded if there
                are not enough proposals.
     gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)]

     Returns: Target ROIs and corresponding class IDs and bounding box shifts.
     rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
     class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
     deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
             Class-specific bbox refinements.

     Note: Returned arrays might be zero padded if not enough target ROIs.
     """
    gt_boxes = torch.cuda.FloatTensor(gt_boxes)
    # Compute overlaps matrix [rpn_rois, gt_boxes]
    # 1. Tile GT boxes and repeat ROIs tensor.
    rois = proposals.unsqueeze(1).repeat(1, 1, gt_boxes.shape[0]).view(-1, 4)
    boxes = gt_boxes.repeat(proposals.shape[0], 1)
    # 2. Compute intersections
    roi_y1, roi_x1, roi_y2, roi_x2 = rois.split(1, dim=1)
    box_y1, box_x1, box_y2, box_x2, class_ids = boxes.split(1, dim=1)
    y1 = torch.max(roi_y1, box_y1)
    x1 = torch.max(roi_x1, box_x1)
    y2 = torch.min(roi_y2, box_y2)
    x2 = torch.min(roi_x2, box_x2)
    zero = torch.cuda.FloatTensor([0])
    intersection = torch.max(x2 - x1, zero) * torch.max(y2 - y1, zero)
    # 3. Compute unions
    roi_area = (roi_y2 - roi_y1) * (roi_x2 - roi_x1)
    box_area = (box_y2 - box_y1) * (box_x2 - box_x1)
    union = roi_area + box_area - intersection
    # 4. Compute IoU and reshape to [rois, boxes]
    iou = intersection / union
    overlaps = iou.view(proposals.shape[0], gt_boxes.shape[0])

    # Determine positive and negative ROIs
    roi_iou_max, _ = torch.max(overlaps, dim=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    positive_indices = torch.nonzero(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box
    negative_indices = torch.nonzero(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 25% positive
    # Positive ROIs
    positive_count = 16
    rand_ids = torch.randperm(positive_indices.shape[0]).cuda()
    positive_indices = positive_indices[rand_ids][:positive_count]
    # Negative ROIs. Fill the rest of the batch.
    negative_count = 64 - positive_indices.shape[0]
    rand_ids = torch.randperm(negative_indices.shape[0]).cuda()
    negative_indices = negative_indices[rand_ids][:negative_count]
    # Gather selected ROIs
    positive_rois = proposals[positive_indices]
    negative_rois = proposals[negative_indices]

    # Assign positive ROIs to GT boxes.
    positive_overlaps = overlaps[positive_indices]
    _, roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)
    roi_gt_boxes = gt_boxes[roi_gt_box_assignment]

    # Compute bbox refinement for positive ROIs
    deltas = box_refinement(positive_rois, roi_gt_boxes[:, :4])

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = torch.cat([positive_rois, negative_rois], dim=0)
    # N = negative_rois.shape[0]
    # P = torch.max(64 - rois.shape[0], 0)
    # rois = F.pad(rois, (0, P))
    # roi_gt_boxes = F.pad(roi_gt_boxes, (0, N+P))
    # deltas = F.pad(deltas, (0, N+P))

    return rois, roi_gt_boxes[:, 4], deltas
