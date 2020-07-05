# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import time
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
from axelerate.networks.common_utils.feature import create_feature_extractor

def create_yolo_network(architecture,
                        input_size,
                        nb_classes,
                        nb_box,
                        weights):
    feature_extractor = create_feature_extractor(architecture, input_size, weights)
    yolo_net = YoloNetwork(feature_extractor,
                           input_size,
                           nb_classes,
                           nb_box)
    return yolo_net


class YoloNetwork(object):
    
    def __init__(self,
                 feature_extractor,
                 input_size,
                 nb_classes,
                 nb_box):
        
        # 1. create full network
        grid_size_x, grid_size_y = feature_extractor.get_output_size()
        
        # make the object detection layer
        output_tensor = Conv2D(nb_box * (4 + 1 + nb_classes), (1,1), strides=(1,1),
                               padding='same', 
                               name='detection_layer_{}'.format(nb_box * (4 + 1 + nb_classes)), 
                               kernel_initializer='lecun_normal')(feature_extractor.feature_extractor.outputs[0])
        output_tensor = Reshape((grid_size_x, grid_size_y, nb_box, 4 + 1 + nb_classes))(output_tensor)
    
        model = Model(feature_extractor.feature_extractor.inputs[0], output_tensor)
        self._norm = feature_extractor.normalize
        self._model = model
        self._init_layer()

    def _init_layer(self):
        layer = self._model.layers[-2]
        weights = layer.get_weights()
        
        input_depth = weights[0].shape[-2] # 2048
        new_kernel = np.random.normal(size=weights[0].shape)/ input_depth
        new_bias   = np.zeros_like(weights[1])

        layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path):
        self._model.load_weights(weight_path)
        
    def forward(self, image):
        netout = self._model.predict(image)[0]
        return netout

    def get_model(self, first_trainable_layer=None):
        return self._model

    def get_grid_size(self):
        _, w, h, _, _ = self._model.get_output_shape_at(-1)
        return (w,h)

    def get_normalize_func(self):
        return self._norm



