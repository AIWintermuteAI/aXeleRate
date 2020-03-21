from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re
import time
import argparse
import cv2
import numpy as np

from box import BoundBox, nms_boxes, boxes_to_array, to_minmax, draw_boxes
from tflite_runtime.interpreter import Interpreter
from flask import Flask, render_template, request, Response
from camera_opencv import Camera

app = Flask (__name__, static_url_path = '')

class Detector(object):
    def __init__(self, label_file, model_file, threshold):
        self._threshold = threshold
        self.labels = self.load_labels(label_file)
        self.interpreter = Interpreter(model_file)
        self._interpreter.allocate_tensors()
        _, self._input_height, self._input_width, _ = self._interpreter.get_input_details()[0]['shape']

    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}


    def set_input_tensor(image):
      """Sets the input tensor."""
      tensor_index = self.interpreter.get_input_details()[0]['index']
      input_tensor = self.interpreter.tensor(tensor_index)()[0]
      input_tensor[:, :] = image

    def get_output_tensor(self, index):
      """Returns the output tensor at the given index."""
      output_details = self.interpreter.get_output_details()[index]
      tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
      return tensor

    def detect_objects(interpreter, image):
      """Returns a list of detection results, each a dictionary of object info."""
      set_input_tensor(interpreter, image)
      interpreter.invoke()
      # Get all output details
      boxes = get_output_tensor(interpreter, 0)
      return boxes

    def detect(self, original_image):
        start_time = time.time()
        image = cv2.resize(original_image, (self.height, self.width), Image.ANTIALIAS)
        results = detect_objects(self._interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000

        def _to_original_scale(boxes):
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= self.width
            minmax_boxes[:,2] *= self.width
            minmax_boxes[:,1] *= self.height
            minmax_boxes[:,3] *= self.height
            return minmax_boxes.astype(np.int)

        boxes, probs = self.run(results)
        boxes = _to_original_scale(boxes)
        image = draw_boxes(image, boxes, probs, self.labels)
        return cv2.imencode('.jpg', image)[1].tobytes()


    def run(self, netout):
        anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        nms_threshold=0.2
        """Convert Yolo network output to bounding box
        
        # Args
            netout : 4d-array, shape of (grid_h, grid_w, num of boxes per grid, 5 + n_classes)
                YOLO neural network output array
        
        # Returns
            boxes : array, shape of (N, 4)
                coordinate scale is normalized [0, 1]
            probs : array, shape of (N, nb_classes)
        """
        grid_h, grid_w, nb_box = netout.shape[:3]
        boxes = []
        
        # decode the output by the network
        netout[..., 4]  = _sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > self._threshold
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                        w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        box = BoundBox(x, y, w, h, confidence, classes)
                        boxes.append(box)
        
        boxes = nms_boxes(boxes, len(classes), nms_threshold, self._threshold)
        boxes, probs = boxes_to_array(boxes)
        return boxes, probs

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

@app.route ("/")
def index ( ):
   return render_template ('index.html', name = None)

def gen(camera):
    while True:
        frame = camera.get_frame()
        image = detector.detect(frame)
        yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
parser.add_argument('--threshold', help='Confidence threshold.', default=0.3)
args = parser.parse_args()

detector = Detector(args.labels, args.model, args.threshold)

if __name__ == "__main__" :
   app.run (host = '0.0.0.0', port = 5000, debug = True)
    
