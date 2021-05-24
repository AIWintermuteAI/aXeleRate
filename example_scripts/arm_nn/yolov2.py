# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Contains functions specific to decoding and processing inference results for YOLO V3 Tiny models.
"""

import cv2
import numpy as np
from box import BoundBox, nms_boxes, boxes_to_array, to_minmax, draw_boxes


def yolo_processing(netout):
    anchors = [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025]
    nms_threshold=0.2
    """Convert Yolo network output to bounding box

    # Args
        netout : 4d-array, shape of (grid_h, grid_w, num of boxes per grid, 5 + n_classes)
            YOLO neural network output array

    # Returns
        boxes : array, shape of (N, 4)
            coordinate scale is normalized [0, 1]
        probs : array, shape of (N, nb_classes)
    """
    netout = netout[0].reshape(7,7,5,6)
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []

    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > 0.3

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    box = BoundBox(x, y, w, h, confidence, classes)
                    boxes.append(box)

    boxes = nms_boxes(boxes, len(classes), nms_threshold, 0.3)
    boxes, probs = boxes_to_array(boxes)
    #print(boxes)
    predictions = []
    def _to_original_scale(boxes):
        minmax_boxes = to_minmax(boxes)
        minmax_boxes[:,0] *= 224
        minmax_boxes[:,2] *= 224
        minmax_boxes[:,1] *= 224
        minmax_boxes[:,3] *= 224
        return minmax_boxes.astype(np.int)

    if len(boxes) > 0:
        boxes = _to_original_scale(boxes)

        for i in range(len(boxes)):
            predictions.append([0, boxes[i], probs[i][0]])

    return predictions

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def yolo_resize_factor(video: cv2.VideoCapture, input_binding_info: tuple):
    """
    Gets a multiplier to scale the bounding box positions to
    their correct position in the frame.

    Args:
        video: Video capture object, contains information about data source.
        input_binding_info: Contains shape of model input layer.

    Returns:
        Resizing factor to scale box coordinates to output frame size.
    """
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    model_height, model_width = list(input_binding_info[1].GetShape())[1:3]
    return max(frame_height, frame_width) / max(model_height, model_width)
