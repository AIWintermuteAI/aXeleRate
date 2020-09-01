# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import keras
import numpy as np
import warnings

from axelerate.networks.yolo.backend.utils.map_evaluation import MapEvaluation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
metrics_dict = {'val_accuracy':['accuracy'],'val_loss':[],'mAP':[]}

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, filepath, metric):

        super(PlotCallback, self).__init__()
        self.filepath = filepath
        self.metric = metric.split("_")[1]
        self.val_metric = metric
        self.values = [0]
        self.val_values = [0]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.values.append(logs.get(self.metric))
        self.val_values.append(logs.get(self.val_metric))
        plot(self.values,self.val_values,self.filepath)

def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         project_folder = 'project',
         first_trainable_layer=None,
         network=None,
         metrics="val_loss"):
    """A function that performs training on a general keras model.

    # Args
        model : keras.models.Model instance
        loss_func : function
            refer to https://keras.io/losses/

        train_batch_gen : keras.utils.Sequence instance
        valid_batch_gen : keras.utils.Sequence instance
        learning_rate : float
        saved_weights_name : str
    """
    # Create project directory
    train_start = time.time()
    train_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(project_folder,train_date)
    basename = network.__class__.__name__ + "_best_"+ metrics
    print('Current training session folder is {}'.format(path))
    os.makedirs(path)
    save_weights_name = os.path.join(path, basename + '.h5')
    save_plot_name = os.path.join(path, basename + '.jpg')
    save_weights_name_ctrlc = os.path.join(path, basename + '_ctrlc.h5')
    print('\n')

    # 1 Freeze layers
    layer_names = [layer.name for layer in model.layers]
    fixed_layers = []
    if first_trainable_layer in layer_names:
        for layer in model.layers:
            if layer.name == first_trainable_layer:
                break
            layer.trainable = False
            fixed_layers.append(layer.name)
    elif not first_trainable_layer:
        pass
    else:
        print('First trainable layer specified in config file is not in the model. Did you mean one of these?')
        for i,layer in enumerate(model.layers):
            print(i,layer.name)
        raise Exception('First trainable layer specified in config file is not in the model')

    if fixed_layers != []:
        print("The following layers do not update weights!!!")
        print("    ", fixed_layers)

    # 2 create optimizer
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer = SGD(lr=learning_rate, nesterov=True)

    # 3. create loss function
    model.compile(loss=loss_func,optimizer=optimizer,metrics=metrics_dict[metrics])
    model.summary()

    #4 create callbacks   
    early_stop = EarlyStopping(monitor=metrics, 
                       min_delta=0.001, 
                       patience=20, 
                       mode='auto', 
                       verbose=1,
                       restore_best_weights=True)
                       
    checkpoint = ModelCheckpoint(save_weights_name, 
                                 monitor=metrics, 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto', 
                                 period=1)
                                 
    reduce_lr = ReduceLROnPlateau(monitor=metrics, factor=0.2,
                              patience=10, min_lr=0.00001,verbose=1)

    map_evaluator_cb = MapEvaluation(network, valid_batch_gen,
                                     save_best=True,
                                     save_name=save_weights_name,
                                     save_plot_name=save_plot_name,
                                     iou_threshold=0.5,
                                     score_threshold=0.3)

    if network.__class__.__name__ == 'YOLO' and metrics =='mAP':
        callbacks = [map_evaluator_cb, ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001,verbose=1)]
    else:
        graph = PlotCallback(save_plot_name,metrics)
        callbacks= [early_stop, checkpoint, reduce_lr, graph] 

    # 4. training
    try:
        model.fit_generator(generator = train_batch_gen,
                        steps_per_epoch  = len(train_batch_gen), 
                        epochs           = nb_epoch,
                        validation_data  = valid_batch_gen,
                        validation_steps = len(valid_batch_gen),
                        callbacks        = callbacks,                        
                        verbose          = 1,
                        workers          = 2,
                        max_queue_size   = 4)
    except KeyboardInterrupt:
        model.save(save_weights_name_ctrlc,overwrite=True,include_optimizer=False)
        return model.layers, save_weights_name_ctrlc 

    _print_time(time.time()-train_start)
    return model.layers, save_weights_name

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

def plot(metric, val_metric, filename):
    plt.figure(figsize=(10,10))
    plt.plot(metric, 'g')
    plt.plot(val_metric, 'r')

    plt.annotate("{:.4f}".format(metric[-1]),xy=(len(metric)-1,metric[-1]))
    plt.annotate("{:.4f}".format(val_metric[-1]),xy=(len(val_metric)-1,val_metric[-1]))

    plt.title('Training graph')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(filename))
    plt.show(block=False)
    plt.pause(1)
    plt.close()

