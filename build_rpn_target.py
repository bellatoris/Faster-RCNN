import numpy as np

from utils import compute_overlaps, box_refinement


def build_rpn_targets(anchors, gt_boxes, rpn_batch_size):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, class_id)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN_match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((rpn_batch_size, 4))

    # Compute overlaps [num_anchors, num_gt_boxes]
    # Each cell contains the IoU of an anchor and GT box.
    overlaps = compute_overlaps(anchors, gt_boxes[:, :4])

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above.
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. It gets overwritten if a gt box is matched
    #    to them.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = np.max(overlaps, axis=1)
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (rpn_batch_size // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (rpn_batch_size - np.sum(rpn_match == 1))
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]  # positive anchors' index
    positive_anchors = anchors[ids]
    gt = gt_boxes[anchor_iou_argmax[ids], :4]  # positive anchor's best matching gt

    rpn_bbox[np.arange(gt.shape[0]), :] = box_refinement(positive_anchors, gt)

    return rpn_match, rpn_bbox
