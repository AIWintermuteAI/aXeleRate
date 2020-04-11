# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import keras
import numpy as np

from axelerate.networks.yolo.backend.utils.map_evaluation import MapEvaluation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import warnings


class CheckpointPB(keras.callbacks.Callback):

    def __init__(self, filepath, date, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CheckpointPB, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.date = date
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.loss = []
        self.val_loss = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        plot(self.loss,self.val_loss,self.filepath,self.date)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(os.path.join(self.filepath,self.date+'.h5'), overwrite=True, include_optimizer=False)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         project_folder = 'project',
         first_trainable_layer=None,
         network=None):
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
    print('Current training session folder is {}'.format(path))
    os.makedirs(path)
    saved_weights_name = os.path.join(path, train_date + '.h5')
    saved_weights_name_ctrlc = os.path.join(path, train_date + '_ctrlc.h5')

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

    # 3. create loss function
    model.compile(loss=loss_func,optimizer=optimizer)
    model.summary()

    #4 create callbacks   
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.001, 
                       patience=20, 
                       mode='min', 
                       verbose=1,
                       restore_best_weights=True)
    checkpoint = CheckpointPB(path,train_date, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001,verbose=1)

    map_evaluator_cb = MapEvaluation(network, valid_batch_gen,
                                     save_best=True,
                                     save_name=saved_weights_name,
                                     iou_threshold=0.7,
                                     score_threshold=0.3)

    if network.__class__.__name__ == 'YOLO':
        callbacks = [map_evaluator_cb,reduce_lr]
    else:
        callbacks= [early_stop, checkpoint, reduce_lr] 



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
        model.save(saved_weights_name_ctrlc,overwrite=True,include_optimizer=False)
        return model.layers, saved_weights_name_ctrlc 

    _print_time(time.time()-train_start)
    return model.layers, saved_weights_name

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

def plot(acc,val_acc,path,train_date):
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, train_date +'.png'))

