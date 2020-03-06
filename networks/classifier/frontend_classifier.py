# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os
import numpy as np

from ..common_utils.feature import create_feature_extractor
from .data_gen import create_datagen
from ..common_utils.fit import train
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

def get_labels(directory):
    labels = os.listdir(directory)
    return labels

def create_classifier(architecture, labels, input_size, layers, dropout ,first_trainable_layer):
    base_model=create_feature_extractor(architecture, input_size, first_trainable_layer)
    x=base_model.feature_extractor.output
    x=GlobalAveragePooling2D()(x)
    for layer in layers[0:-1]:
        x=Dense(layer,activation='relu')(x) 
        x=Dropout(dropout)(x)
    x=Dense(layers[-1],activation='relu')(x)
    preds=Dense(len(labels),activation='softmax')(x)
    model=Model(inputs=base_model.feature_extractor.input,outputs=preds)
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
            self._yolo_network.load_weights(weight_path, by_name=by_name)
        else:
            print("Fail to load pre-trained weights. Make sure weight file path.")

    def predict(self, image):
        preprocessed_image = prepare_image(image,show=False)
        pred = model.predict(preprocessed_image)
        predicted_class_indices=np.argmax(pred,axis=1)
        predictions = [labels[k] for k in predicted_class_indices]
        return predictions

    def train(self,
              img_folder,
              nb_epoch,
              saved_weights_name,
              batch_size=8,
              jitter=True,
              learning_rate=1e-4, 
              train_times=1,
              valid_times=1,
              valid_img_folder="",
              first_trainable_layer=None):
        
        train_generator, validation_generator = create_datagen(img_folder, valid_img_folder, batch_size, self._input_size, saved_weights_name)
        self._network.summary()
        train(self._network,'categorical_crossentropy',train_generator,validation_generator,learning_rate, nb_epoch,saved_weights_name)
        print("Saving model")
        self._network.save(saved_weights_name)
    
