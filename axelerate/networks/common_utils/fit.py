import shutil
import os
import time
import tensorflow as tf
import numpy as np
import warnings

from axelerate.networks.common_utils.callbacks import WarmUpCosineDecayScheduler
from axelerate.networks.yolo.backend.utils.custom import MergeMetrics
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         project_folder = 'project',
         first_trainable_layer = None,
         metric=None,
         metric_name="val_loss"):
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
    path = os.path.join(project_folder, train_date)
    basename = model.name + "_best_"+ metric_name
    print('Current training session folder is {}'.format(path))
    os.makedirs(path)
    save_weights_name = os.path.join(path, basename + '.h5')
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
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    if not metric:
        metric = metric_name
    else:
        metric = metric[metric_name]

    print(metric)    
    # 3. create loss function
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metric if metric != 'loss' else None)
    model.summary()

    #4 create callbacks   
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("logs", histogram_freq=1)
    
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
                                            total_steps=len(train_batch_gen)*nb_epoch,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=len(train_batch_gen)*min(3, nb_epoch-1),
                                            hold_base_rate_steps=0,
                                            verbose=1)

    if metric_name in ['recall', 'precision']:
        mergedMetric = MergeMetrics(model, metric_name, 1, True, save_weights_name, tensorboard_callback)
        callbacks = [mergedMetric, warm_up_lr, tensorboard_callback]  
    else:  
        early_stop = EarlyStopping(monitor='val_' + metric, 
                                min_delta=0.001, 
                                patience=20, 
                                mode='auto', 
                                verbose=2,
                                restore_best_weights=True)
                       
        checkpoint = ModelCheckpoint(save_weights_name, 
                                 monitor='val_' + metric, 
                                 verbose=2, 
                                 save_best_only=True, 
                                 mode='auto', 
                                 period=1)
                                 
        reduce_lr = ReduceLROnPlateau(monitor='val_' + metric,
                                factor=0.2,
                                patience=10,
                                min_lr=1e-6,
                                mode='auto',
                                verbose=2)   
        callbacks = [early_stop, checkpoint, warm_up_lr, tensorboard_callback] 

    # 4. training
    try:
        model.fit(train_batch_gen,
                steps_per_epoch  = len(train_batch_gen), 
                epochs           = nb_epoch,
                validation_data  = valid_batch_gen,
                validation_steps = len(valid_batch_gen),
                callbacks        = callbacks,                        
                verbose          = 1,
                workers          = 4,
                max_queue_size   = 10,
                use_multiprocessing = True)
    except KeyboardInterrupt:
        print("Saving model and copying logs")
        model.save(save_weights_name_ctrlc, overwrite=True, include_optimizer=False)
        shutil.copytree("logs", os.path.join(path, "logs"))  
        return model.layers, save_weights_name_ctrlc 
        
    shutil.copytree("logs", os.path.join(path, "logs"))
    _print_time(time.time()-train_start)
    return model.layers, save_weights_name

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

