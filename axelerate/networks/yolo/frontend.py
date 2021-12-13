# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from axelerate.networks.common_utils.fit import train
from axelerate.networks.yolo.backend.decoder import YoloDecoder
from axelerate.networks.yolo.backend.utils.custom import Yolo_Precision, Yolo_Recall
from axelerate.networks.yolo.backend.loss import create_loss_fn, Params
from axelerate.networks.yolo.backend.network import create_yolo_network
from axelerate.networks.yolo.backend.batch_gen import create_batch_generator
from axelerate.networks.yolo.backend.utils.annotation import get_train_annotations, get_unique_labels
from axelerate.networks.yolo.backend.utils.box import to_minmax

def get_object_labels(ann_directory):
    files = os.listdir(ann_directory)
    files = [os.path.join(ann_directory, fname) for fname in files]
    return get_unique_labels(files)

def create_yolo(architecture,
                labels,
                input_size,
                anchors,
                obj_thresh,
                iou_thresh,
                coord_scale,
                object_scale,
                no_object_scale,
                weights = None):

    n_classes = len(labels)
    n_boxes = int(len(anchors[0]))
    n_branches = len(anchors)
    yolo_network = create_yolo_network(architecture, input_size, n_classes, n_boxes, n_branches, weights)
    yolo_params = Params(obj_thresh, iou_thresh, object_scale, no_object_scale, coord_scale, yolo_network.get_grid_size(), anchors, n_classes)
    yolo_loss = create_loss_fn

    metrics_dict = {'recall': [Yolo_Precision(obj_thresh, name='precision'), Yolo_Recall(obj_thresh, name='recall')],
                    'precision': [Yolo_Precision(obj_thresh, name='precision'), Yolo_Recall(obj_thresh, name='recall')]}

    yolo_decoder = YoloDecoder(anchors, yolo_params, 0.1, input_size)
    yolo = YOLO(yolo_network, yolo_loss, yolo_decoder, labels, input_size, yolo_params, metrics_dict)
    return yolo


class YOLO(object):
    def __init__(self,
                 yolo_network,
                 yolo_loss,
                 yolo_decoder,
                 labels,
                 input_size,
                 yolo_params,
                 metrics_dict):

        self.yolo_network = yolo_network
        self.yolo_loss = yolo_loss
        self.yolo_decoder = yolo_decoder
        self.labels = labels
        self.input_size = input_size
        self.norm = yolo_network._norm
        self.yolo_params = yolo_params
        self.num_branches = len(self.yolo_params.anchors)
        self.metrics_dict = metrics_dict

    def load_weights(self, weight_path, by_name=True):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights for the whole model: ", weight_path)
            self.yolo_network.load_weights(weight_path, by_name=True)
        else:
            print("Failed to load pre-trained weights for the whole model. It might be because you didn't specify any or the weight file cannot be found")

    def predict(self, image, height, width, threshold=0.3):
        """
        # Args
            image : 3d-array (RGB ordered)
        
        # Returns
            boxes : array, shape of (N, 4)
            probs : array, shape of (N, nb_classes)
        """

        def _to_original_scale(boxes):
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= width
            minmax_boxes[:,2] *= width
            minmax_boxes[:,1] *= height
            minmax_boxes[:,3] *= height
            return minmax_boxes.astype(np.int)

        start_time = time.time()
        netout = self.yolo_network.forward(image)
        elapsed_ms = (time.time() - start_time) * 1000
        boxes, probs= self.yolo_decoder.run(netout, threshold)

        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            print(boxes, probs)
            return elapsed_ms, boxes, probs
        else:
            return elapsed_ms, [], []

    def evaluate(self, img_folder, ann_folder, batch_size):

        self.generator = create_batch_generator(img_folder, ann_folder, self.input_size, 
                                                self.output_size, self.n_classes, 
                                                batch_size, 1, False, self.norm)
        tp = np.zeros(self.n_classes)
        fp = np.zeros(self.n_classes)
        fn = np.zeros(self.n_classes)
        n_pixels = np.zeros(self.n_classes)
        
        for inp, gt in tqdm(list(self.generator)):
            y_pred = self.network.predict(inp)        

    def train(self,
              img_folder,
              ann_folder,
              nb_epoch,
              project_folder,
              batch_size,
              jitter,
              learning_rate, 
              train_times,
              valid_times,
              valid_img_folder,
              valid_ann_folder,
              first_trainable_layer,
              metrics):

        # 1. get annotations        
        train_annotations, valid_annotations = get_train_annotations(self.labels,
                                                                     img_folder,
                                                                     ann_folder,
                                                                     valid_img_folder,
                                                                     valid_ann_folder,
                                                                     is_only_detect = False)
        # 1. get batch generator
        valid_batch_size = len(valid_annotations)*valid_times
        if valid_batch_size < batch_size: 
            raise ValueError("Not enough validation images: batch size {} is larger than {} validation images. Add more validation images or decrease batch size!".format(batch_size, valid_batch_size))
        
        train_batch_generator = self._get_batch_generator(train_annotations, batch_size, train_times, augment=jitter)
        valid_batch_generator = self._get_batch_generator(valid_annotations, batch_size, valid_times, augment=False)
        
        # 2. To train model get keras model instance & loss function
        model = self.yolo_network.get_model(first_trainable_layer)
        loss = self._get_loss_func(batch_size)
        
        # 3. Run training loop
        return train(model,
                loss,
                train_batch_generator,
                valid_batch_generator,
                learning_rate = learning_rate, 
                nb_epoch  = nb_epoch,
                project_folder = project_folder,
                first_trainable_layer = first_trainable_layer,
                metric=self.metrics_dict,
                metric_name=metrics)

    def _get_loss_func(self, batch_size):
        return [self.yolo_loss(self.yolo_params, layer, batch_size) for layer in range(self.num_branches)]

    def _get_batch_generator(self, annotations, batch_size, repeat_times, augment):
        """
        # Args
            annotations : Annotations instance
            batch_size : int
            jitter : bool
        
        # Returns
            batch_generator : BatchGenerator instance
        """
        batch_generator = create_batch_generator(annotations,
                                                 self.input_size,
                                                 self.yolo_network.get_grid_size(),
                                                 batch_size,
                                                 self.yolo_params.anchors,
                                                 repeat_times,
                                                 augment=augment,
                                                 norm=self.yolo_network.get_normalize_func())
        return batch_generator
    
