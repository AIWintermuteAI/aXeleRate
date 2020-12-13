import argparse
import io
import time
import numpy as np
import cv2

from box import BoundBox, nms_boxes, boxes_to_array, to_minmax, draw_boxes
#from tflite_runtime.interpreter import Interpreter
import tflite_runtime.interpreter as tflite

class Detector(object):

    def __init__(self, label_file, model_file, threshold):
        self._threshold = float(threshold)
        self.labels = self.load_labels(label_file)
        self.interpreter = tflite.Interpreter(model_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']

    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.read().replace('"','').split(','))}

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32)
        img = img / 255.
        img = img - 0.5
        img = img * 2.
        img = img[:, :, ::-1]
        img = np.expand_dims(img, 0)
        return img

    def get_output_tensor(self, index):
      """Returns the output tensor at the given index."""
      output_details = self.interpreter.get_output_details()[index]
      tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
      return tensor

    def detect_objects(self, image):
      """Returns a list of detection results, each a dictionary of object info."""
      img = self.preprocess(image)
      self.interpreter.set_tensor(self.tensor_index, img)
      self.interpreter.invoke()
      # Get all output details
      raw_detections = self.get_output_tensor(0)
      output_shape = [7, 7, 5, 6]
      output = np.reshape(raw_detections, output_shape)
      return output 

    def detect(self, original_image):
        self.output_height, self.output_width = original_image.shape[0:2]
        start_time = time.time()
        results = self.detect_objects(original_image)
        elapsed_ms = (time.time() - start_time) * 1000
        fps  = 1 / elapsed_ms*1000
        print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))

        def _to_original_scale(boxes):
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= self.output_width
            minmax_boxes[:,2] *= self.output_width
            minmax_boxes[:,1] *= self.output_height
            minmax_boxes[:,3] *= self.output_height
            return minmax_boxes.astype(np.int)

        boxes, probs = self.run(results)
        print(boxes)
        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            original_image = draw_boxes(original_image, boxes, probs, self.labels)
        return original_image


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


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
parser.add_argument('--threshold', help='Confidence threshold.', default=0.3)
args = parser.parse_args()

detector = Detector(args.labels, args.model, args.threshold)
camera = cv2.VideoCapture(2)

while(camera.isOpened()):
    ret, frame = camera.read()
    image = detector.detect(frame)
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', image)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
camera.release()

# Closes all the frames
cv2.destroyAllWindows()
