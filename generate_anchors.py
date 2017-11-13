"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [128, 256, 512]
    ratios: 1D array of anchor ratios of width / height. Example: [2, 1, 0.5]
    shape: [height, width] spatial shape of the feature map over which
           to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors of the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    widths = scales * np.sqrt(ratios)
    heights = scales / np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_center_x = np.meshgrid(widths, shifts_x)
    box_heights, box_center_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_center_y, box_center_x],
                           axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths],
                         axis=2).reshape([-1, 2])

    # Convert to corner coordinate (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes


if __name__ == '__main__':
    scales = [128, 256, 512]
    ratios = [2, 1, 0.5]

    feature_stride = 16
    anchor_stride = 1

    boxes = generate_anchors(scales, ratios, [38, 50],
                             feature_stride, anchor_stride)
