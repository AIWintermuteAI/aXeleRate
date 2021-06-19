import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow import map_fn
import numpy as np
import os
import skimage
import cv2
from math import cos, sin

def tf_xywh_to_all(grid_pred_xy, grid_pred_wh, layer, params):
    """ rescale the pred raw [grid_pred_xy,grid_pred_wh] to [0~1]

    Parameters
    ----------
    grid_pred_xy : tf.Tensor

    grid_pred_wh : tf.Tensor

    layer : int
        the output layer
    h : Helper


    Returns
    -------
    tuple

        after process, [all_pred_xy, all_pred_wh] 
    """
    with tf.name_scope('xywh_to_all_%d' % layer):
        #print('xyoffset', params.xy_offset[layer], 'outhw', params.out_hw[layer][::-1])
        all_pred_xy = (tf.sigmoid(grid_pred_xy[..., :]) + params.xy_offset[layer]) / params.out_hw[layer][::-1]
        all_pred_wh = tf.exp(grid_pred_wh[..., :]) * params.anchors[layer]
    return all_pred_xy, all_pred_wh


def tf_xywh_to_grid(all_true_xy, all_true_wh, layer, params):
    """convert true label xy wh to grid scale

    Parameters
    ----------
    all_true_xy : tf.Tensor

    all_true_wh : tf.Tensor

    layer : int
        layer index
    h : Helper


    Returns
    -------
    [tf.Tensor, tf.Tensor]
        grid_true_xy, grid_true_wh shape = [out h ,out w,anchor num , 2 ]
    """
    with tf.name_scope('xywh_to_grid_%d' % layer):
        grid_true_xy = (all_true_xy * params.out_hw[layer][::-1]) - params.xy_offset[layer]
        grid_true_wh = tf.math.log(all_true_wh / params.anchors[layer])
    return grid_true_xy, grid_true_wh


def tf_reshape_box(true_xy_A: tf.Tensor, true_wh_A: tf.Tensor, p_xy_A: tf.Tensor, p_wh_A: tf.Tensor, layer: int, params) -> tuple:
    """ reshape the xywh to [?,h,w,anchor_nums,true_box_nums,2]
        NOTE  must use obj mask in atrue xywh !
    Parameters
    ----------
    true_xy_A : tf.Tensor
        shape will be [true_box_nums,2]

    true_wh_A : tf.Tensor
        shape will be [true_box_nums,2]

    p_xy_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]

    p_wh_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]

    layer : int

    helper : Helper


    Returns
    -------
    tuple
        true_cent, true_box_wh, pred_cent, pred_box_wh
    """
    with tf.name_scope('reshape_box_%d' % layer):
        true_cent = true_xy_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        true_box_wh = true_wh_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]

        true_cent = tf.tile(true_cent, [helper.batch_size, helper.out_hw[layer][0], helper.out_hw[layer][1], helper.anchor_number, 1, 1])
        true_box_wh = tf.tile(true_box_wh, [helper.batch_size, helper.out_hw[layer][0], helper.out_hw[layer][1], helper.anchor_number, 1, 1])

        pred_cent = p_xy_A[..., tf.newaxis, :]
        pred_box_wh = p_wh_A[..., tf.newaxis, :]
        pred_cent = tf.tile(pred_cent, [1, 1, 1, 1, tf.shape(true_xy_A)[0], 1])
        pred_box_wh = tf.tile(pred_box_wh, [1, 1, 1, 1, tf.shape(true_wh_A)[0], 1])

    return true_cent, true_box_wh, pred_cent, pred_box_wh


def tf_iou(pred_xy: tf.Tensor, pred_wh: tf.Tensor, vaild_xy: tf.Tensor, vaild_wh: tf.Tensor) -> tf.Tensor:
    """ calc the iou form pred box with vaild box

    Parameters
    ----------
    pred_xy : tf.Tensor
        pred box shape = [out h, out w, anchor num, 2]

    pred_wh : tf.Tensor
        pred box shape = [out h, out w, anchor num, 2]

    vaild_xy : tf.Tensor
        vaild box shape = [? , 2]

    vaild_wh : tf.Tensor
        vaild box shape = [? , 2]

    Returns
    -------
    tf.Tensor
        iou value shape = [out h, out w, anchor num ,?]
    """
    b1_xy = tf.expand_dims(pred_xy, -2)
    b1_wh = tf.expand_dims(pred_wh, -2)
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = tf.expand_dims(vaild_xy, 0)
    b2_wh = tf.expand_dims(vaild_wh, 0)
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def calc_ignore_mask(t_xy_A: tf.Tensor, t_wh_A: tf.Tensor, p_xy: tf.Tensor, p_wh: tf.Tensor, obj_mask: tf.Tensor, iou_thresh: float, layer: int, params) -> tf.Tensor:
    """clac the ignore mask

    Parameters
    ----------
    t_xy_A : tf.Tensor
        raw ture xy,shape = [batch size,h,w,anchors,2]
    t_wh_A : tf.Tensor
        raw true wh,shape = [batch size,h,w,anchors,2]
    p_xy : tf.Tensor
        raw pred xy,shape = [batch size,h,w,anchors,2]
    p_wh : tf.Tensor
        raw pred wh,shape = [batch size,h,w,anchors,2]
    obj_mask : tf.Tensor
        old obj mask,shape = [batch size,h,w,anchors]
    iou_thresh : float
        iou thresh 
    helper : Helper
        Helper obj

    Returns
    -------
    tf.Tensor
    ignore_mask : 
        ignore_mask, shape = [batch size, h, w, anchors, 1]
    """
    with tf.name_scope('calc_mask_%d' % layer):
        pred_xy, pred_wh = tf_xywh_to_all(p_xy, p_wh, layer, params)

        ignore_mask = []
        for bc in range(params.batch_size):
            vaild_xy = tf.boolean_mask(t_xy_A[bc], obj_mask[bc])
            vaild_wh = tf.boolean_mask(t_wh_A[bc], obj_mask[bc])
            iou_score = tf_iou(pred_xy[bc], pred_wh[bc], vaild_xy, vaild_wh)
            best_iou = tf.reduce_max(iou_score, axis=-1, keepdims=True)
            ignore_mask.append(tf.cast(best_iou < iou_thresh, tf.float32))
    return tf.stack(ignore_mask)


class Params:

    def __init__(self, obj_thresh, iou_thresh, obj_weight, noobj_weight, wh_weight, out_hw, anchors, class_num):
        self.obj_thresh = obj_thresh
        self.iou_thresh = iou_thresh
        self.wh_weight = wh_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.class_num = class_num
        self.out_hw = np.reshape(np.array(out_hw), (-1, 2))
        #print(self.out_hw)
        self.anchors = anchors

        self.grid_wh = (1 / self.out_hw)[:, [1, 0]]
        #print(self.grid_wh)
        self.wh_scale = Params._anchor_scale(self.anchors, self.grid_wh)
        self.xy_offset = Params._coordinate_offset(self.anchors, self.out_hw)

        self.batch_size = None

    @staticmethod
    def _coordinate_offset(anchors: np.ndarray, out_hw: np.ndarray) -> np.array:
        """construct the anchor coordinate offset array , used in convert scale

        Parameters
        ----------
        anchors : np.ndarray
            anchors shape = [n,] = [ n x [m,2]]
        out_hw : np.ndarray
            output height width shape = [n,2]

        Returns
        -------
        np.array
            scale shape = [n,] = [n x [h_n,w_n,m,2]]
        """
        grid = []
        for l in range(len(anchors)):
            grid_y = np.tile(np.reshape(np.arange(0, stop=out_hw[l][0]), [-1, 1, 1, 1]), [1, out_hw[l][1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=out_hw[l][1]), [1, -1, 1, 1]), [out_hw[l][0], 1, 1, 1])
            grid.append(np.concatenate([grid_x, grid_y], axis=-1))
        return np.array(grid)

    @staticmethod
    def _anchor_scale(anchors: np.ndarray, grid_wh: np.ndarray) -> np.array:
        """construct the anchor scale array , used in convert label to annotation

        Parameters
        ----------
        anchors : np.ndarray
            anchors shape = [n,] = [ n x [m,2]]
        out_hw : np.ndarray
            output height width shape = [n,2]

        Returns
        -------
        np.array
            scale shape = [n,] = [n x [m,2]]
        """
        return np.array([anchors[i] * grid_wh[i] for i in range(len(anchors))])


def create_loss_fn(params, layer, batch_size):

    params.batch_size = batch_size
    shapes = [[-1] + list(params.out_hw[layer]) + [len(params.anchors[layer]), params.class_num + 5]]
    #print(shapes)
    # @tf.function
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
        #print(y_true, y_pred)
        """ split the label """
        grid_pred_xy = y_pred[..., 0:2]
        grid_pred_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 4:5]
        pred_cls = y_pred[..., 5:]

        all_true_xy = y_true[..., 0:2]
        all_true_wh = y_true[..., 2:4]
        true_confidence = y_true[..., 4:5]
        true_cls = y_true[..., 5:]

        obj_mask = true_confidence  # true_confidence[..., 0] > obj_thresh
        obj_mask_bool = y_true[..., 4] > params.obj_thresh

        """ calc the ignore mask  """

        ignore_mask = calc_ignore_mask(all_true_xy, all_true_wh, grid_pred_xy,
                                       grid_pred_wh, obj_mask_bool,
                                       params.iou_thresh, layer, params)

        grid_true_xy, grid_true_wh = tf_xywh_to_grid(all_true_xy, all_true_wh, layer, params)
        # NOTE When wh=0 , tf.log(0) = -inf, so use K.switch to avoid it
        grid_true_wh = K.switch(obj_mask_bool, grid_true_wh, tf.zeros_like(grid_true_wh))

        """ define loss """
        coord_weight = 2 - all_true_wh[..., 0:1] * all_true_wh[..., 1:2]

        xy_loss = tf.reduce_sum(
            obj_mask * coord_weight * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=grid_true_xy, logits=grid_pred_xy)) / params.batch_size

        wh_loss = tf.reduce_sum(
            obj_mask * coord_weight * params.wh_weight * tf.square(tf.subtract(
                x=grid_true_wh, y=grid_pred_wh))) / params.batch_size

        obj_loss = params.obj_weight * tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence)) / params.batch_size

        noobj_loss = params.noobj_weight * tf.reduce_sum(
            (1 - obj_mask) * ignore_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_confidence, logits=pred_confidence)) / params.batch_size

        cls_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_cls, logits=pred_cls)) / params.batch_size

        total_loss = obj_loss + noobj_loss + cls_loss + xy_loss + wh_loss

        return total_loss

    return loss_fn