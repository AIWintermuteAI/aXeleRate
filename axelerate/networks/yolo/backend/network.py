# -*- coding: utf-8 -*-
from functools import reduce, wraps

import numpy as np
import tensorflow as tf
from axelerate.networks.common_utils.feature import create_feature_extractor
from axelerate.networks.common_utils.mobilenet_sipeed.mobilenet import (
    _conv_block, _depthwise_conv_block)
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     LeakyReLU, Reshape, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model


def create_yolo_network(architecture, input_size, nb_classes, nb_box, nb_stages, weights):
    feature_extractor = create_feature_extractor(architecture, input_size, weights)
    if architecture in ["MobileNet1_0", "MobileNet7_5", "MobileNet5_0", "MobileNet2_5"]:
        yolo_net = YOLOMobile(feature_extractor, nb_stages, nb_classes, nb_box)
    elif architecture in ["MobileNetV2_1_0", "MobileNetV2_7_5", "MobileNetV2_5_0", "MobileNetV2_2_5"]:
        yolo_net = YOLOMobileV2(feature_extractor,
                                nb_stages,
                                nb_classes,
                                nb_box,
                                alpha=float(architecture[-3:].replace('_', '.')))
    else:
        raise NotImplementedError("%s backbone not supported yet!" % (architecture))

    return yolo_net


class YOLOBase(object):
    def _init_layers(self, layers):
        for layer in layers:
            layer = self._model.get_layer(layer)
            weights = layer.get_weights()

            input_depth = weights[0].shape[-2]  # 2048
            new_kernel = np.random.normal(size=weights[0].shape) / input_depth
            new_bias = np.zeros_like(weights[1])

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


class YOLOMobile(YOLOBase):
    def __init__(self, feature_extractor, nb_stages, nb_classes, nb_box):

        # 1. create full network
        grid_size_y, grid_size_x = feature_extractor.get_output_size(layer='conv_pw_13_relu')
        x1 = feature_extractor.get_output_tensor('conv_pw_13_relu')

        # make the object detection layer
        y1 = Conv2D(nb_box * (4 + 1 + nb_classes), (1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='detection_layer_1',
                    kernel_initializer='lecun_normal')(x1)

        if nb_stages == 2:
            grid_size_y_2, grid_size_x_2 = feature_extractor.get_output_size(layer='conv_pw_11_relu')
            x2 = feature_extractor.get_output_tensor('conv_pw_11_relu')
            #x1 = _depthwise_conv_block(inputs = x1, alpha = 1, pointwise_conv_filters = 128, block_id=14)
            x1 = UpSampling2D(2)(x1)

            if x1.shape[1:3] != x2.shape[1:3]:
                #print(x1.shape[1:3] - x2.shape[1:3])
                #pad = tf.math.subtract(x1.shape[1:3], x2.shape[1:3]).numpy().tolist()
                #print(pad)
                x2 = ZeroPadding2D(padding=((0, 1), (0, 0)))(x2)
                grid_size_y_2, grid_size_x_2 = x2.shape[1:3]

            x2 = Concatenate()([x2, x1])
            #x2 = _depthwise_conv_block(inputs = x2, alpha = 1, pointwise_conv_filters = 128, block_id=14)

            y2 = Conv2D(nb_box * (4 + 1 + nb_classes), (1, 1),
                        strides=(1, 1),
                        padding='same',
                        name='detection_layer_2',
                        kernel_initializer='lecun_normal')(x2)

        if nb_stages == 2:

            l1 = Reshape((grid_size_y, grid_size_x, nb_box, 4 + 1 + nb_classes))(y1)
            l2 = Reshape((grid_size_y_2, grid_size_x_2, nb_box, 4 + 1 + nb_classes))(y2)

            detection_layers = ['detection_layer_1', 'detection_layer_2']
            output_tensors = [l1, l2]
        else:

            l1 = Reshape((grid_size_y, grid_size_x, nb_box, 4 + 1 + nb_classes))(y1)

            detection_layers = ['detection_layer_1']
            output_tensors = [l1]

        model = Model(feature_extractor.feature_extractor.inputs[0], output_tensors, name='yolo')
        self._norm = feature_extractor.normalize
        self._model = model
        self._init_layers(detection_layers)


class YOLOMobileV2(YOLOBase):
    def __init__(self, feature_extractor, nb_stages, nb_classes, nb_box, **kwargs):

        # 1. create full network
        grid_size_y, grid_size_x = feature_extractor.get_output_size(layer='block_16_expand_relu')
        x1 = feature_extractor.get_output_tensor('block_16_expand_relu')

        # make the object detection layer
        y1 = compose(DarknetConv2D_BN_Leaky(128 if kwargs['alpha'] > 0.7 else 192, (3, 3)),
                     DarknetConv2D(nb_box * (nb_classes + 5), (1, 1), name='detection_layer_1'))(x1)

        if nb_stages == 2:
            grid_size_y_2, grid_size_x_2 = feature_extractor.get_output_size(layer='block_13_expand_relu')
            x2 = feature_extractor.get_output_tensor('block_13_expand_relu')

            x1 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x1)
            if x1.shape[1:3] != x2.shape[1:3]:
                #print(x1.shape[1:3] - x2.shape[1:3])
                #pad = tf.math.subtract(x1.shape[1:3], x2.shape[1:3]).numpy().tolist()
                #print(pad)
                x2 = ZeroPadding2D(padding=((0, 1), (0, 0)))(x2)
                grid_size_y_2, grid_size_x_2 = x2.shape[1:3]

            y2 = compose(Concatenate(),
                         DarknetConv2D_BN_Leaky(128 if kwargs['alpha'] > 0.7 else 192, (3, 3)),
                         DarknetConv2D(nb_box * (nb_classes + 5), (1, 1), name='detection_layer_2'))([x1, x2])

        if nb_stages == 2:
            l1 = Reshape((grid_size_y, grid_size_x, nb_box, 4 + 1 + nb_classes))(y1)
            l2 = Reshape((grid_size_y_2, grid_size_x_2, nb_box, 4 + 1 + nb_classes))(y2)

            detection_layers = ['detection_layer_1', 'detection_layer_2']
            output_tensors = [l1, l2]
        else:
            l1 = Reshape((grid_size_y, grid_size_x, nb_box, 4 + 1 + nb_classes))(y1)

            detection_layers = ['detection_layer_1']
            output_tensors = [l1]

        model = Model(feature_extractor.feature_extractor.inputs[0], output_tensors, name='yolo')
        self._norm = feature_extractor.normalize
        self._model = model
        self._init_layers(detection_layers)


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))
