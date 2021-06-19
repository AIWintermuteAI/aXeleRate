import numpy as np
from axelerate.networks.yolo.backend.utils.box import BoundBox
from axelerate.networks.yolo.backend.utils.box import BoundBox, nms_boxes, boxes_to_array

class YoloDecoder(object):
    
    def __init__(self,
                 anchors,
                 params,
                 nms_threshold,
                 input_size):

        self.anchors = anchors
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.params = params

    def run(self, netout, obj_threshold):
        boxes = []

        for l, output in enumerate(netout):
            output = np.squeeze(output)
            grid_h, grid_w, nb_box = output.shape[0:3]
            
            # decode the output by the network
            output[..., 4] = _sigmoid(output[..., 4])
            output[..., 5:] = output[..., 4][..., np.newaxis] * _sigmoid(output[..., 5:])
            output[..., 5:] *= output[..., 5:] > obj_threshold
            
            for row in range(grid_h):
                for col in range(grid_w):
                    for b in range(nb_box):
                        # from 4th element onwards are confidence and class classes
                        classes = output[row, col, b, 5:]

                        if np.sum(classes) > 0:
                            # first 4 elements are x, y, w, and h
                            x, y, w, h = output[row, col, b, :4]

                            x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                            y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                            w = self.anchors[l][b][0] * np.exp(w) # unit: image width
                            h = self.anchors[l][b][1] * np.exp(h) # unit: image height
                            confidence = output[row, col, b, 4]
                            box = BoundBox(x, y, w, h, confidence, classes)
                            boxes.append(box)

        boxes = nms_boxes(boxes, len(classes), self.nms_threshold, obj_threshold)
        boxes, probs = boxes_to_array(boxes)

        return boxes, probs

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

