import time
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from axelerate.networks.common_utils.feature import create_feature_extractor
from axelerate.networks.classifier.batch_gen import create_datagen
from axelerate.networks.common_utils.fit import train
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet import preprocess_input

def get_labels(directory):
    labels = sorted(os.listdir(directory))
    return labels

def create_classifier(architecture, labels, input_size, layers, dropout, weights = None, save_bottleneck = False):
    base_model = create_feature_extractor(architecture, input_size, weights)
    x = base_model.feature_extractor.outputs[0]
    x = GlobalAveragePooling2D()(x)
    if len(layers) != 0:
        for layer in layers[0:-1]:
            x = Dense(layer, activation = 'relu')(x) 
            x = Dropout(dropout)(x)
        x = Dense(layers[-1], activation = 'relu')(x)
    preds = Dense(len(labels), activation = 'softmax')(x)
    model = Model(inputs = base_model.feature_extractor.inputs[0],outputs = preds, name = 'classifier')

    bottleneck_layer = None
    if save_bottleneck:
        bottleneck_layer = base_model.feature_extractor.layers[-1].name
    network = Classifier(model, input_size, labels, base_model.normalize, bottleneck_layer)

    return network

class Classifier(object):
    def __init__(self,
                 network,
                 input_size,
                 labels,
                 norm,
                 bottleneck_layer):
        self.network = network       
        self.labels = labels
        self.input_size = input_size
        self.bottleneck_layer = bottleneck_layer
        self.norm = norm

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights for the whole model: ", weight_path)
            self.network.load_weights(weight_path)
        else:
            print("Failed to load pre-trained weights for the whole model. It might be because you didn't specify any or the weight file cannot be found")

    def save_bottleneck(self, model_path, bottleneck_layer):
        bottleneck_weights_path = os.path.join(os.path.dirname(model_path),'bottleneck_weigths.h5')
        model = load_model(model_path)
        for layer in model.layers:
            if layer.name == bottleneck_layer:
                output = layer.output
        bottleneck_model = Model(model.input, output)
        bottleneck_model.save_weights(bottleneck_weights_path)

    def predict(self, img):

        start_time = time.time()
        Y_pred = np.squeeze(self.network(img, training = False))
        elapsed_ms = (time.time() - start_time)  * 1000

        y_pred = np.argmax(Y_pred)
        prob = Y_pred[y_pred]

        prediction = self.labels[y_pred]

        return elapsed_ms, prob, prediction

    def evaluate(self, img_folder, batch_size):

        self.generator = create_datagen(img_folder, batch_size, self.input_size, None, False, self.norm)

        Y_pred = self.network.predict(self.generator, len(self.generator) // batch_size + 1)

        y_pred = np.argmax(Y_pred, axis=1)

        print('Classification Report')
        report = classification_report(self.generator.classes, y_pred, target_names = self.labels)
        print(report)

        print('Confusion Matrix')
        cm = confusion_matrix(self.generator.classes, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = self.labels)
        disp.plot(include_values=True, cmap='Blues', ax=None)
        plt.show()

        return report, cm

    def train(self,
              img_folder,
              nb_epoch,
              project_folder,
              batch_size = 8,
              augumentation = False,
              learning_rate = 1e-4, 
              train_times = 1,
              valid_times = 1,
              valid_img_folder = "",
              first_trainable_layer = None,
              metrics = "val_loss"):

        if metrics != "accuracy" and metrics != "loss":
            print("Unknown metric for Classifier, valid options are: val_loss or val_accuracy. Defaulting ot val_loss")
            metrics = "loss"

        train_generator = create_datagen(img_folder, batch_size, self.input_size, project_folder, augumentation, self.norm)
        validation_generator = create_datagen(valid_img_folder, batch_size, self.input_size, project_folder, False, self.norm)

        model_layers, model_path = train(self.network,
                                        'categorical_crossentropy',
                                        train_generator,
                                        validation_generator,
                                        learning_rate, 
                                        nb_epoch, 
                                        project_folder,
                                        first_trainable_layer, 
                                        metric_name = metrics)

        if self.bottleneck_layer:
            self.save_bottleneck(model_path, self.bottleneck_layer)
        return model_layers, model_path

    
