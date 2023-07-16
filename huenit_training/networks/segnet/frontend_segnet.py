import os
import numpy as np
import cv2
import time
from tqdm import tqdm

from axelerate.networks.segnet.data_utils.data_loader import create_batch_generator, verify_segmentation_dataset
from axelerate.networks.common_utils.feature import create_feature_extractor
from axelerate.networks.common_utils.fit import train
from axelerate.networks.segnet.models.segnet import mobilenet_segnet, squeezenet_segnet, full_yolo_segnet, tiny_yolo_segnet, nasnetmobile_segnet, resnet50_segnet, densenet121_segnet

def masked_categorical_crossentropy(gt , pr ):
    from tensorflow.keras.losses import categorical_crossentropy
    mask = 1 -  gt[: , : , 0] 
    return categorical_crossentropy(gt, pr)*mask

def create_segnet(architecture, input_size, n_classes, weights = None):

    if architecture == 'NASNetMobile':
        model = nasnetmobile_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif architecture == 'SqueezeNet':
        model = squeezenet_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif architecture == 'Full Yolo':
        model = full_yolo_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif architecture == 'Tiny Yolo':
        model = tiny_yolo_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif architecture == 'DenseNet121':
        model = densenet121_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif architecture == 'ResNet50':
        model = resnet50_segnet(n_classes, input_size, encoder_level=4, weights = weights)
    elif 'MobileNet' in architecture:
        model = mobilenet_segnet(n_classes, input_size, encoder_level=4, weights = weights, architecture = architecture)

    output_size = (model.output_height, model.output_width)
    network = Segnet(model, input_size, n_classes, model.normalize, output_size)

    return network

class Segnet(object):
    def __init__(self,
                 network,
                 input_size,
                 n_classes,
                 norm,
                 output_size):
        self.network = network       
        self.n_classes = n_classes
        self.input_size = input_size
        self.output_size = output_size
        self.norm = norm

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights for the whole model: ", weight_path)
            self.network.load_weights(weight_path)
        else:
            print("Failed to load pre-trained weights for the whole model. It might be because you didn't specify any or the weight file cannot be found")

    def predict(self, image):

        start_time = time.time()
        Y_pred = np.squeeze(self.network.predict(image))
        elapsed_ms = (time.time() - start_time)  * 1000

        y_pred = np.argmax(Y_pred, axis = 2)

        return elapsed_ms, y_pred


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

                y_pred = np.argmax(y_pred, axis=-1)
                gt = np.argmax(gt, axis=-1)

                for cl_i in range(self.n_classes):
                    
                    tp[cl_i] += np.sum((y_pred == cl_i) * (gt == cl_i))
                    fp[cl_i] += np.sum((y_pred == cl_i) * ((gt != cl_i)))
                    fn[cl_i] += np.sum((y_pred != cl_i) * ((gt == cl_i)))
                    n_pixels[cl_i] += np.sum(gt == cl_i)

        cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
        n_pixels_norm = n_pixels /  np.sum(n_pixels)
        frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
        mean_IU = np.mean(cl_wise_score)
        report = {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU":cl_wise_score}
        return report

    def train(self,
              img_folder,
              ann_folder,
              nb_epoch,
              project_folder,
              batch_size=8,
              do_augment=False,
              learning_rate=1e-4, 
              train_times=1,
              valid_times=1,
              valid_img_folder="",
              valid_ann_folder="",
              first_trainable_layer=None,
              ignore_zero_class=False,
              metrics='val_loss'):
        
        if metrics != "accuracy" and metrics != "loss":
            print("Unknown metric for SegNet, valid options are: val_loss or val_accuracy. Defaulting ot val_loss")
            metrics = "loss"

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'
        train_generator = create_batch_generator(img_folder, ann_folder, self.input_size, 
                          self.output_size, self.n_classes,batch_size, train_times, do_augment, self.norm)

        validation_generator = create_batch_generator(valid_img_folder, valid_ann_folder, self.input_size, 
                               self.output_size, self.n_classes, batch_size, valid_times, False, self.norm)
        
        return train(self.network,
                            loss_k,
                            train_generator, 
                            validation_generator, 
                            learning_rate, 
                            nb_epoch, 
                            project_folder, 
                            first_trainable_layer, 
                            metric_name = metrics)
    
