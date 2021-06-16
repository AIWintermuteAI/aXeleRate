import numpy as np
from axelerate.networks.yolo.backend.utils.box import BoundBox
from axelerate.networks.yolo.backend.utils.box import BoundBox, nms_boxes, boxes_to_array
import tensorflow as tf
from tensorflow.python import keras
from axelerate.networks.yolo.backend.loss import tf_xywh_to_all


def correct_box(box_xy, box_wh, input_shape, image_shape):
    """rescale predict box to original image scale

    Parameters
    ----------
    box_xy : tf.Tensor
        box xy
    box_wh : tf.Tensor
        box wh
    input_shape : list
        input shape
    image_shape : list
        image shape

    Returns
    -------
    tf.Tensor
        new boxes
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


class YoloDecoder(object):
    
    def __init__(self,
                 anchors,
                 params,
                 nms_threshold,
                 input_size):

        self.anchors = anchors
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.params = params

    def run(self, netout, obj_threshold, orig_size):

        """ box list """
        _yxyx_box = []
        _yxyx_box_scores = []

        #print(netout.shape)
        """ preprocess label """
        for l, pred_label in enumerate(netout):
            """ split the label """
            pred_xy = pred_label[..., 0:2]
            pred_wh = pred_label[..., 2:4]
            pred_confidence = pred_label[..., 4:5]
            pred_cls = pred_label[..., 5:]

            box_scores = tf.sigmoid(pred_cls) * tf.sigmoid(pred_confidence)

            """ reshape box  """
            # NOTE tf_xywh_to_all will auto use sigmoid function
            pred_xy_A, pred_wh_A = tf_xywh_to_all(pred_xy, pred_wh, l, self.params)
            boxes = correct_box(pred_xy_A, pred_wh_A, self.input_size, orig_size)
            boxes = tf.reshape(boxes, (-1, 4))
            box_scores = tf.reshape(box_scores, (-1, self.params.class_num))
            """ append box and scores to global list """
            _yxyx_box.append(boxes)
            _yxyx_box_scores.append(box_scores)

            yxyx_box = tf.concat(_yxyx_box, axis=0)
            yxyx_box_scores = tf.concat(_yxyx_box_scores, axis=0)

            mask = yxyx_box_scores >= obj_threshold
            #print(mask.shape)
            """ do nms for every classes"""
            _boxes = []
            _scores = []
            _classes = []

            for c in range(self.params.class_num):
                class_boxes = tf.boolean_mask(yxyx_box, mask[:, c])
                class_box_scores = tf.boolean_mask(yxyx_box_scores[:, c], mask[:, c])
                select = tf.image.non_max_suppression(
                    class_boxes, scores=class_box_scores, max_output_size=30, iou_threshold=self.nms_threshold)
                class_boxes = tf.gather(class_boxes, select)
                class_box_scores = tf.gather(class_box_scores, select)
                _boxes.append(class_boxes)
                _scores.append(class_box_scores)
                _classes.append(tf.ones_like(class_box_scores) * c)

            boxes = tf.concat(_boxes, axis=0).numpy()
            classes = tf.concat(_classes, axis=0).numpy()
            scores = tf.concat(_scores, axis=0).numpy()

            if len(classes) > 0:
                print(f'[top\tleft\tbottom\tright\tscore\tclass]')
                for i, c in enumerate(classes):
                    box = boxes[i]
                    score = scores[i]
                    top, left, bottom, right = box
                    print(f'[{top:.1f}\t{left:.1f}\t{bottom:.1f}\t{right:.1f}\t{score:.2f}\t{int(c):2d}]')
                    top = max(0, np.floor(top + 0.5))
                    left = max(0, np.floor(left + 0.5))
                    bottom = min(orig_size[0], np.floor(bottom + 0.5))
                    right = min(orig_size[1], np.floor(right + 0.5))
                    boxes[i] = [top, left, bottom, right]
                    
        boxes = boxes.astype(np.int64)
        classes = classes.astype(np.int)
        return boxes, scores, classes


