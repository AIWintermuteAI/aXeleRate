import tensorflow as tf
import numpy as np

from tensorflow import keras
import tensorflow_model_optimization as tfmot

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

def prune(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate,
         nb_epoch,
         model_path = 'project',
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
    project_folder = os.path.dirname(model_path)
    basename = model.name + "_best_" + metric_name + "_pruned"
    print('Current training session folder is {}'.format(project_folder))

    save_weights_name = os.path.join(project_folder, basename + '.h5')
    save_weights_name_ctrlc = os.path.join(project_folder, basename + '_ctrlc.h5')
    print('\n')

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after after 5 epochs.
    end_epoch = 5

    num_iterations_per_epoch = len(train_batch_gen)
    end_step = num_iterations_per_epoch * end_epoch

    # Define parameters for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                                final_sparsity=0.75,
                                                                begin_step=0,
                                                                end_step=end_step),
        'pruning_policy': tfmot.sparsity.keras.PruneForLatencyOnXNNPack()
    }

    # Try to apply pruning wrapper with pruning policy parameter.
    try:
        model = prune_low_magnitude(model, **pruning_params)
    except ValueError as e:
        print(e)


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

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir="logs")]

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
        
    return model.layers, save_weights_name

def train_qat(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate,
         nb_epoch,
         model_path = 'project',
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
    project_folder = os.path.dirname(model_path)
    basename = model.name + "_best_" + metric_name + "_qat"
    print('Current training session folder is {}'.format(project_folder))

    save_weights_name = os.path.join(project_folder, basename + '.h5')
    save_weights_name_ctrlc = os.path.join(project_folder, basename + '_ctrlc.h5')
    print('\n')

    quantize_model = tfmot.quantization.keras.quantize_model
    model = quantize_model(model)

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

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir="logs")]

    # 4. training
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

    print("Saving model and copying logs")
    model.save(save_weights_name, overwrite=True, include_optimizer=False)
    shutil.copytree("logs", os.path.join(project_folder, "logs"))  
        
    return model.layers, save_weights_name