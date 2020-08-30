import os
import tensorflow as tf
import numpy as np
import keras
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MapEvaluation(keras.callbacks.Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, yolo, generator,
                 iou_threshold=0.5,
                 score_threshold=0.3,
                 save_path=None,
                 save_plot_name=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None):
                 
        super().__init__()
        self._yolo = yolo
        self._generator = generator
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._save_path = save_path
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._tensorboard = tensorboard

        self.loss = [0]
        self.val_loss = [0]
        self.maps = [0]
        self._save_plot_name = save_plot_name

        self.bestMap = 0

        if not isinstance(self._tensorboard, keras.callbacks.TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if epoch % self._period == 0 and self._period != 0:
            _map, average_precisions = self.evaluate_map()
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self._yolo._labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(_map))

            if epoch == 0:
                print("Saving model on first epoch irrespective of mAP")
                self.model.save(self._save_name,overwrite=True,include_optimizer=False)
            else:
                if self._save_best and self._save_name is not None and _map > self.bestMap:
                    print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap, _map, self._save_name))
                    self.bestMap = _map
                    self.model.save(self._save_name,overwrite=True,include_optimizer=False)
                else:
                    print("mAP did not improve from {}.".format(self.bestMap))


            self.loss.append(logs.get("loss"))
            self.val_loss.append(logs.get("val_loss"))
            self.maps.append(_map)
            plot(self.loss,self.val_loss,self.maps,self._save_plot_name)

            if self._tensorboard is not None and self._tensorboard.writer is not None:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = _map
                summary_value.tag = "val_mAP"
                self._tensorboard.writer.add_summary(summary, epoch)

    def evaluate_map(self):
        average_precisions = self._calc_avg_precisions()
        _map = sum(average_precisions.values()) / len(average_precisions)

        return _map, average_precisions

    def _calc_avg_precisions(self):

        # gather all detections and annotations
        all_detections = [[None for _ in range(len(self._yolo._labels))]
                          for _ in range(len(self._generator)*self._generator._batch_size)]
        all_annotations = [[None for _ in range(len(self._yolo._labels))]
                           for _ in range(len(self._generator)*self._generator._batch_size)]

        counter = 0
        for i in range(len(self._generator)):
            img_batch, annotations = self._generator.load_batch(i)
            for j in range(len(annotations)):
                raw_image = img_batch[j]
                height, width = raw_image.shape[:2]
                input_image = np.expand_dims(raw_image, 0)
                # make the boxes and the labels
                _, pred_boxes, probs = self._yolo.predict(input_image, height, width, threshold=self._score_threshold)

                if len(pred_boxes) > 0:
                    score = np.array(probs)  
                    pred_labels = np.argmax(score,axis=1)
                else:
                    pred_boxes = np.array([[]])
                    score = np.array(probs) 
                    pred_labels = score
                   
                # sort the boxes and the labels according to scores
                #score_sort = np.argsort(-score)
                #pred_labels = pred_labels[score_sort]
                #pred_boxes = pred_boxes[score_sort]
                    
                # copy detections to all_detections
                for label in range(len(self._yolo._labels)):
                    all_detections[counter][label] = pred_boxes[pred_labels == label, :]

                # copy ground truth to all_annotations
                for label in range(len(self._yolo._labels)):
                    all_annotations[counter][label] = annotations[j][annotations[j][:, 4] == label, :4].copy()
                counter += 1

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(len(self._yolo._labels)):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(counter):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []
                
                for d in detections:
                    #scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue
                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]
                    if max_overlap >= self._iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            #indices = np.argsort(-scores)
            #false_positives = false_positives[indices]
            #true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def plot(acc, val_acc, maps, filename):
    plt.figure(figsize=(10,10))
    plt.plot(acc, 'g')
    plt.plot(val_acc, 'r')
    plt.plot(maps, 'b')

    plt.annotate("{:.4f}".format(acc[-1]),xy=(len(acc)-1,acc[-1]))
    plt.annotate("{:.4f}".format(val_acc[-1]),xy=(len(val_acc)-1,val_acc[-1]))
    plt.annotate("{:.4f}".format(maps[-1]),xy=(len(maps)-1,maps[-1]))

    plt.title('Training graph')
    plt.ylabel('Loss, mAP')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test', 'mAP'], loc='upper left')
    plt.savefig(os.path.join(filename))
    
    plt.show(block=False)
    plt.pause(1)
    plt.close()

