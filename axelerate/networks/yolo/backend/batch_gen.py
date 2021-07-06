import cv2
import os
import numpy as np
np.random.seed(1337)

from tensorflow.keras.utils import Sequence
from axelerate.networks.common_utils.augment import ImgAugment
from axelerate.networks.yolo.backend.utils.box import to_centroid, create_anchor_boxes, find_match_box
from axelerate.networks.common_utils.fit import train


def create_batch_generator(annotations, 
                           input_size,
                           grid_sizes,
                           batch_size,
                           anchors,
                           repeat_times,
                           augment, 
                           norm=None):
    """
    # Args
        annotations : Annotations instance in utils.annotation module
    
    # Return 
        worker : BatchGenerator instance
    """

    img_aug = ImgAugment(input_size[0], input_size[1], augment)
    yolo_box = _YoloBox(input_size, grid_sizes)
    netin_gen = _NetinGen(input_size, norm)
    netout_gen = _NetoutGen(grid_sizes, annotations.n_classes(), anchors)
    worker = BatchGenerator(netin_gen,
                            netout_gen,
                            yolo_box,
                            img_aug,
                            annotations,
                            batch_size,
                            repeat_times)
    return worker


class BatchGenerator(Sequence):
    def __init__(self,
                 netin_gen,
                 netout_gen,
                 yolo_box,
                 img_aug,
                 annotations,
                 batch_size,
                 repeat_times):
        """
        # Args
            annotations : Annotations instance

        """
        self._netin_gen = netin_gen
        self._netout_gen = netout_gen
        self.nb_stages = len(netout_gen.anchors)
        self._img_aug = img_aug
        self._yolo_box = yolo_box

        self._batch_size = min(batch_size, len(annotations)*repeat_times)
        self._repeat_times = repeat_times
        self.annotations = annotations
        self.counter = 0

    def __len__(self):
        return int(len(self.annotations) * self._repeat_times /self._batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        x_batch = []
        y_batch1 = []

        if self.nb_stages == 2:
            y_batch2 = []

        for i in range(self._batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self._batch_size*idx + i)
            boxes = self.annotations.boxes(self._batch_size*idx + i)
            labels = self.annotations.code_labels(self._batch_size*idx + i)

            # 2. read image in fixed size
            img, boxes, labels = self._img_aug.imread(fname, boxes, labels)

            # 3. grid scaling centroid boxes
            if len(boxes) > 0:
                norm_boxes = self._yolo_box.trans(boxes)
            else:
                norm_boxes = []
                labels = []
      
            # 4. generate x_batch
            x_batch.append(self._netin_gen.run(img))
            processed_labels = self._netout_gen.run(norm_boxes, labels)

            y_batch1.append(processed_labels[0])
            if self.nb_stages == 2:           
                y_batch2.append(processed_labels[1])

        x_batch = np.array(x_batch)
        y_batch1 = np.array(y_batch1)
        batch = y_batch1

        if self.nb_stages == 2:           
            y_batch2 = np.array(y_batch2)
            batch = [y_batch1, y_batch2]

        self.counter += 1
        return x_batch, batch

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0

class _YoloBox(object):

    def __init__(self, input_size, grid_size):
        self._input_size = input_size
        self._grid_size = grid_size

    def trans(self, boxes):
        """
        # Args
            boxes : array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered & input image size scale coordinate

        # Returns
            norm_boxes : array, same shape of boxes
                (cx, cy, w, h)-ordered & rescaled to grid-size
        """
        # 1. [[100, 120, 140, 200]] minimax box -> centroid box
        centroid_boxes = to_centroid(boxes).astype(np.float32)
        # 2. [[120. 160.  40.  80.]] image scale -> imga scle 0 ~ 1 [[4.        5.        1.3333334 2.5      ]]
        norm_boxes = np.zeros_like(centroid_boxes)
        norm_boxes[:,0::2] = centroid_boxes[:,0::2] / self._input_size[1]
        norm_boxes[:,1::2] = centroid_boxes[:,1::2] / self._input_size[0]
        #print("norm boxes", norm_boxes)
        return norm_boxes

class _NetinGen(object):
    def __init__(self, input_size, norm):
        self._input_size = input_size
        self._norm = self._set_norm(norm)

    def run(self, image):
        return self._norm(image)

    def _set_norm(self, norm):
        if norm is None:
            return lambda x: x
        else:
            return norm

class _NetoutGen(object):
    def __init__(self,
                 grid_sizes,
                 nb_classes,
                 anchors):
        self.nb_classes = nb_classes
        self.anchors = np.asarray(anchors)
        self._tensor_shape = self._set_tensor_shape(grid_sizes, nb_classes)

    def run(self, norm_boxes, labels):
        """
        # Args
            norm_boxes : array, shape of (N, 4)
                scale normalized boxes
            labels : list of integers
            y_shape : tuple (grid_size, grid_size, nb_boxes, 4+1+nb_classes)
        """
        labels = np.asarray([labels])
        norm_boxes = np.asarray(norm_boxes)
        if len(norm_boxes) > 0:
            norm_boxes= np.concatenate((labels.T, norm_boxes), axis = 1)
        #print("boxes", boxes)
        y = self.box_to_label(norm_boxes)
        #print(y.shape)

        return y

    def _set_tensor_shape(self, grid_size, nb_classes):
        nb_boxes = len(self.anchors[0])
        return [(grid_size[i][0], grid_size[i][1], nb_boxes, 4+1+nb_classes) for i in range(len(self.anchors))]

    def _xy_grid_index(self, box_xy: np.ndarray, layer: int):
        """ get xy index in grid scale

        Parameters
        ----------
        box_xy : np.ndarray
            value = [x,y]
        layer  : int
            layer index

        Returns
        -------
        [np.ndarray,np.ndarray]

            index xy : = [idx,idy]
        """
        out_wh = self._tensor_shape[layer][0:2:][::-1]
        #print(box_xy, out_wh)
        return np.floor(box_xy * out_wh).astype('int')

    @staticmethod
    def _fake_iou(a: np.ndarray, b: np.ndarray) -> float:
        """set a,b center to same,then calc the iou value

        Parameters
        ----------
        a : np.ndarray
            array value = [w,h]
        b : np.ndarray
            array value = [w,h]

        Returns
        -------
        float
            iou value
        """
        a_maxes = a / 2.
        a_mins = -a_maxes

        b_maxes = b / 2.
        b_mins = -b_maxes

        iner_mins = np.maximum(a_mins, b_mins)
        iner_maxes = np.minimum(a_maxes, b_maxes)
        iner_wh = np.maximum(iner_maxes - iner_mins, 0.)
        iner_area = iner_wh[..., 0] * iner_wh[..., 1]

        s1 = a[..., 0] * a[..., 1]
        s2 = b[..., 0] * b[..., 1]

        return iner_area / (s1 + s2 - iner_area)

    def _get_anchor_index(self, wh: np.ndarray) -> np.ndarray:
        """get the max iou anchor index

        Parameters
        ----------
        wh : np.ndarray
            value = [w,h]

        Returns
        -------
        np.ndarray
            max iou anchor index
            value  = [layer index , anchor index]
        """
        iou = _NetoutGen._fake_iou(wh, self.anchors)
        return np.unravel_index(np.argmax(iou), iou.shape)

    def box_to_label(self, true_box: np.ndarray) -> tuple:
        """convert the annotation to yolo v3 label~

        Parameters
        ----------
        true_box : np.ndarray
            annotation shape :[n,5] value :[n*[p,x,y,w,h]]

        Returns
        -------
        tuple
            labels list value :[output_number*[out_h,out_w,anchor_num,class+5]]
        """
        labels = [np.zeros((self._tensor_shape[i][0], self._tensor_shape[i][1], len(self.anchors[i]),
                            5 + self.nb_classes), dtype='float32') for i in range(len(self.anchors))]
        for box in true_box:
            # NOTE box [x y w h] are relative to the size of the entire image [0~1]
            l, n = self._get_anchor_index(box[3:5])  # [layer index, anchor index]
            idx, idy = self._xy_grid_index(box[1:3], l)  # [x index , y index]
            labels[l][idy, idx, n, 0:4] = np.clip(box[1:5], 1e-8, 1.)
            labels[l][idy, idx, n, 4] = 1.
            labels[l][idy, idx, n, 5 + int(box[0])] = 1.

        return labels
