# -*- coding: utf-8 -*-
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Conv2D, UpSampling2D, Concatenate
from axelerate.networks.common_utils.feature import create_feature_extractor
from axelerate.networks.common_utils.mobilenet_sipeed.mobilenet import _depthwise_conv_block, _conv_block

def create_yolo_network(architecture,
                        input_size,
                        nb_classes,
                        nb_box,
                        nb_stages,
                        weights):
    feature_extractor = create_feature_extractor(architecture, input_size, weights)
    yolo_net = YoloNetwork(feature_extractor,
                           nb_stages,
                           nb_classes,
                           nb_box)
    return yolo_net


class YoloNetwork(object):
    
    def __init__(self,
                 feature_extractor,
                 nb_stages,
                 nb_classes,
                 nb_box):
        
        output_tensors = []
        detection_layers = []

        # 1. create full network
        grid_size_y, grid_size_x = feature_extractor.get_output_size(layer  = 'conv_pw_13_relu')
        x1 = feature_extractor.get_output_tensor('conv_pw_13_relu')

        # make the object detection layer
        y1 = Conv2D(nb_box * (4 + 1 + nb_classes), (1,1), strides=(1,1),
                            padding='same', 
                            name='detection_layer_1', 
                            kernel_initializer='lecun_normal')(x1)

        l1 = Reshape((grid_size_y, grid_size_x, nb_box, 4 + 1 + nb_classes))(y1)  
        output_tensors.append(l1)
        detection_layers.append('detection_layer_1')

        if nb_stages == 2:
            grid_size_y, grid_size_x = feature_extractor.get_output_size(layer = 'conv_pw_11_relu')
            x2 = feature_extractor.get_output_tensor('conv_pw_11_relu')
            #x1 = _depthwise_conv_block(inputs = x1, alpha = 1, pointwise_conv_filters = 128, block_id=14)
            x1 = UpSampling2D(2)(x1)
            x2 = Concatenate()([x1, x2])
            #x1 = _depthwise_conv_block(inputs = x1, alpha = 1, pointwise_conv_filters = 128, block_id=14)

            y2 = Conv2D(nb_box * (4 + 1 + nb_classes), (1,1), strides=(1,1),
                                padding='same', 
                                name='detection_layer_2', 
                                kernel_initializer='lecun_normal')(x2)

            l2 = Reshape((14, 14, nb_box, 4 + 1 + nb_classes))(y2)
            output_tensors.append(l2)
            detection_layers.append('detection_layer_2')

        model = Model(feature_extractor.feature_extractor.inputs[0], output_tensors, name='yolo')
        self._norm = feature_extractor.normalize
        self._model = model
        self._init_layers(detection_layers)

    def _init_layers(self, layers):
        for layer in layers:
            layer = self._model.get_layer(layer)
            weights = layer.get_weights()
            
            input_depth = weights[0].shape[-2] # 2048
            new_kernel = np.random.normal(size=weights[0].shape)/ input_depth
            new_bias   = np.zeros_like(weights[1])

            layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path, by_name):
        self._model.load_weights(weight_path, by_name=by_name)
        
    def forward(self, image):
        netout = self._model.predict(image)
        return netout

    def get_model(self, first_trainable_layer=None):
        return self._model

    def get_grid_size(self):
        grid_sizes = []
        for model_output in self._model.outputs:
            grid_sizes.append(list(model_output.shape[1:3]))
        return grid_sizes

    def get_normalize_func(self):
        return self._norm



