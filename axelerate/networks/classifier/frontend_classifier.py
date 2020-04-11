# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.
import time
import os
import numpy as np

from axelerate.networks.common_utils.feature import create_feature_extractor
from axelerate.networks.classifier.data_gen import create_datagen
from axelerate.networks.common_utils.fit import train
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

def prepare_image(file, show=False, size=(224,224)):
    img = image.load_img(file, target_size=size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_labels(directory):
    labels = sorted(os.listdir(directory))
    return labels

def create_classifier(architecture, labels, input_size, layers, dropout):
    base_model=create_feature_extractor(architecture, input_size)
    x=base_model.feature_extractor.outputs[0]
    x=GlobalAveragePooling2D()(x)
    for layer in layers[0:-1]:
        x=Dense(layer,activation='relu')(x) 
        x=Dropout(dropout)(x)
    x=Dense(layers[-1],activation='relu')(x)
    preds=Dense(len(labels),activation='softmax')(x)
    model=Model(inputs=base_model.feature_extractor.inputs[0],outputs=preds)
    network = Classifier(model,input_size,labels)

    return network

class Classifier(object):
    def __init__(self,
                 network,
                 input_size,
                 labels):
        self._network = network       
        self._labels = labels
        self._input_size = input_size

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights in", weight_path)
            self._network.load_weights(weight_path, by_name=by_name)
        else:
            print("Fail to load pre-trained weights-starting training from scratch")

    def predict(self, image):
        preprocessed_image = prepare_image(image,show=False,size=(self._input_size,self._input_size))
        start_time = time.time()
        pred = self._network.predict(preprocessed_image)
        elapsed_ms = (time.time() - start_time) * 1000
        predicted_class_indices=np.argmax(pred,axis=1)
        predictions = [self._labels[k] for k in predicted_class_indices]
        return elapsed_ms, predictions, pred[0][predicted_class_indices]

    def train(self,
              img_folder,
              nb_epoch,
              project_folder,
              batch_size=8,
              augumentation=False,
              learning_rate=1e-4, 
              train_times=1,
              valid_times=1,
              valid_img_folder="",
              first_trainable_layer=None):
        
        train_generator, validation_generator = create_datagen(img_folder, valid_img_folder, batch_size, self._input_size, project_folder, augumentation)
        return train(self._network,'categorical_crossentropy',train_generator,validation_generator,learning_rate, nb_epoch,project_folder,first_trainable_layer)

    
