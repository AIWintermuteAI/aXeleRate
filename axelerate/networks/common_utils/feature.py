import keras
from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.applications import DenseNet121
from keras.applications import NASNetMobile
from keras.applications import ResNet50

from .mobilenet_sipeed.mobilenet import MobileNet

def create_feature_extractor(architecture, input_size, weights = None):
    """
    # Args
        architecture : str
        input_size : int

    # Returns
        feature_extractor : BaseFeatureExtractor instance
    """
    if architecture == 'DenseNet121':
        feature_extractor = DenseNet121Feature(input_size, weights)
    elif architecture == 'SqueezeNet':
        feature_extractor = SqueezeNetFeature(input_size, weights)
    elif architecture == 'MobileNet1_0':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=1)
    elif architecture == 'MobileNet7_5':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.75)
    elif architecture == 'MobileNet5_0':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.5)
    elif architecture == 'MobileNet2_5':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.25)
    elif architecture == 'Full Yolo':
        feature_extractor = FullYoloFeature(input_size, weights)
    elif architecture == 'Tiny Yolo':
        feature_extractor = TinyYoloFeature(input_size, weights)
    elif architecture == 'NASNetMobile':
        feature_extractor = NASNetMobileFeature(input_size, weights)
    elif architecture == 'ResNet50':
        feature_extractor = ResNet50Feature(input_size, weights)
    else:
        raise Exception('Architecture not supported! Name should be Full Yolo, Tiny Yolo, MobileNet1_0, MobileNet7_5, MobileNet5_0, MobileNet2_5, SqueezeNet, NASNetMobile, ResNet50 or DenseNet121')
    return feature_extractor



class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_input_size(self):
        input_shape = self.feature_extractor.get_input_shape_at(0)
        assert input_shape[1] == input_shape[2]
        return input_shape[1]

    def get_output_size(self):
        output_shape = self.feature_extractor.get_output_shape_at(-1)
        return output_shape[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class FullYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights=None):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)

        if weights == 'imagenet':
            print('Imagenet for YOLO backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)

    def normalize(self, image):
        return image / 255.

class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(24*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(312, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)

        if weights == 'imagenet':
            print('Imagenet for YOLO backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)


    def normalize(self, image):
        return image / 255.

class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights, alpha):
        input_image = Input(shape=(input_size[0], input_size[1], 3))
        input_shapes_imagenet = [(128, 128,3), (160, 160,3), (192, 192,3), (224, 224,3)]
        input_shape =(128,128,3)
        for item in input_shapes_imagenet:
            if item[0] <= input_size[0]:
                input_shape = item

        if weights == 'imagenet':
            mobilenet = MobileNet(input_shape=input_shape, input_tensor=input_image, alpha = alpha, weights = 'imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            print('Successfully loaded imagenet backend weights')
        else:
            mobilenet = MobileNet(input_shape=(input_size[0],input_size[1],3),alpha = alpha,depth_multiplier = 1, dropout = 0.001, weights = None, include_top=False, backend=keras.backend, layers=keras.layers,models=keras.models,utils=keras.utils)
            if weights:
                print('Loaded backend weigths: '+weights)
                mobilenet.load_weights(weights)

        #x = mobilenet(input_image)
        self.feature_extractor = mobilenet

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image		

class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x)  
        
        if weights == 'imagenet':
            print('Imagenet for SqueezeNet backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)


    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image    

class DenseNet121Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        if weights == 'imagenet':
            densenet = DenseNet121(input_tensor=input_image, include_top=False, weights='imagenet', pooling=None)
            print('Successfully loaded imagenet backend weights')
        else:
            densenet = DenseNet121(input_tensor=input_image, include_top=False, weights=None, pooling=None)
            if weights:
                densenet.load_weights(weights)
                print('Loaded backend weigths: ' + weights)

        self.feature_extractor = densenet

    def normalize(self, image):
        from keras.applications.densenet import preprocess_input
        return preprocess_input(image)

class NASNetMobileFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        if weights == 'imagenet':
            nasnetmobile = NASNetMobile(input_tensor=input_image, include_top=False, weights='imagenet', pooling=None)
            print('Successfully loaded imagenet backend weights')
        else:
            nasnetmobile = NASNetMobile(input_tensor=input_image, include_top=False, weights=None, pooling=None)
            if weights:
                nasnetmobile.load_weights(weights)
                print('Loaded backend weigths: ' + weights)
        self.feature_extractor = nasnetmobile

    def normalize(self, image):
        from keras.applications.nasnet import preprocess_input
        return preprocess_input(image)

class ResNet50Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        if weights == 'imagenet':
            resnet50 = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling = None)
            print('Successfully loaded imagenet backend weights')
        else:
            resnet50 = ResNet50(input_tensor=input_image, include_top=False, pooling = None)
            if weights:
                resnet50.load_weights(weights)
                print('Loaded backend weigths: ' + weights)

        self.feature_extractor = resnet50

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image 
