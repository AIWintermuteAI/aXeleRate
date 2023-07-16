from tensorflow.python import keras
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras

class Yolo_Precision(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None):
        super(Yolo_Precision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds

        default_threshold = 0.5

        self.thresholds = default_threshold if thresholds is None else thresholds

        self.true_positives = self.add_weight(
            'tp', initializer=init_ops.zeros_initializer)  # type: ResourceVariable

        self.false_positives = self.add_weight(
            'fp', initializer=init_ops.zeros_initializer)  # type: ResourceVariable

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_confidence = y_true[..., 4:5]
        pred_confidence = y_pred[..., 4:5]
        pred_confidence_sigmoid = math_ops.sigmoid(pred_confidence)

        values = math_ops.logical_and(true_confidence > self.thresholds, pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.true_positives.assign_add(math_ops.reduce_sum(values))

        values = math_ops.logical_and(math_ops.logical_not(true_confidence > self.thresholds),
                                      pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.false_positives.assign_add(math_ops.reduce_sum(values))

    def result(self):
        return math_ops.div_no_nan(self.true_positives, (math_ops.add(self.true_positives, self.false_positives)))


class Yolo_Recall(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None):
        super(Yolo_Recall, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds

        default_threshold = 0.5

        self.thresholds = default_threshold if thresholds is None else thresholds

        self.true_positives = self.add_weight(
            'tp', initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'fn', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_confidence = y_true[..., 4:5]
        pred_confidence = y_pred[..., 4:5]
        pred_confidence_sigmoid = math_ops.sigmoid(pred_confidence)

        values = math_ops.logical_and(true_confidence > self.thresholds, pred_confidence > self.thresholds)
        values = math_ops.cast(values, self.dtype)
        self.true_positives.assign_add(math_ops.reduce_sum(values))  # type: ResourceVariable

        values = math_ops.logical_and(true_confidence > self.thresholds,
                                      math_ops.logical_not(pred_confidence > self.thresholds))
        values = math_ops.cast(values, self.dtype)
        self.false_negatives.assign_add(math_ops.reduce_sum(values))  # type: ResourceVariable

    def result(self):
        return math_ops.div_no_nan(self.true_positives, (math_ops.add(self.true_positives, self.false_negatives)))

class MergeMetrics(tensorflow.keras.callbacks.Callback):

    def __init__(self, 
                 model,
                 type,
                 period = 1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None):
                 
        super().__init__()
        self.type = type
        self.name = "total_val_" + self.type
        output_names = []

        for layer in model.layers:
            if 'reshape' in layer.name:
                output_names.append(layer.name)

        self.output_names = ['val_' + output_name + "_" + self.type if len(output_names) > 1 else 'val_' + self.type for output_name in output_names]
        print("Layers to use in {} callback monitoring: {}".format(self.name, self.output_names))

        self.num_outputs = len(self.output_names)
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._tensorboard = tensorboard

        self.best_result = 0

        if not isinstance(self._tensorboard, tensorflow.keras.callbacks.TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if epoch % self._period == 0 and self._period != 0:
            result = sum([logs[output_name] for output_name in self.output_names])/self.num_outputs
            logs[self.name] = result

            print('\n')
            print('{}: {:.4f}'.format(self.name, result))

            if epoch == 0:
                print("Saving model on first epoch irrespective of {}".format(self.name))
                self.model.save(self._save_name, overwrite=True, include_optimizer=False)
            else:
                if self._save_best and self._save_name is not None and result > self.best_result:
                    print("{} improved from {} to {}, saving model to {}.".format(self.name, self.best_result, result, self._save_name))
                    self.best_result = result
                    self.model.save(self._save_name, overwrite=True, include_optimizer=False)
                else:
                    print("{} did not improve from {}.".format(self.name, self.best_result))

            if self._tensorboard:
                writer = tf.summary.create_file_writer(self._tensorboard.log_dir)
                with writer.as_default():
                    tf.summary.scalar(self.name, result, step=epoch)
                    writer.flush()