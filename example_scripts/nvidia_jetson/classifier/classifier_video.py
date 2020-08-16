from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2

import tensorrt as trt
import engine as eng
import inference as inf

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

class Classifier:

  def __init__(self, label_file, model_file):
    self.labels = self.load_labels(label_file)
    self.engine = eng.load_engine(trt_runtime, model_file)
    self.h_input, self.d_input, self.h_output, self.d_output, self.stream = inf.allocate_buffers(self.engine, 1, trt.float32)
    self.context = self.engine.create_execution_context()
    self.width=224
    self.height=224

  def preprocess(self, img):
      image = img.astype(np.float32)
      image = image / 255.
      image = image - 0.5
      image = image * 2.
      image = image[:, :, ::-1]
      return image

  def load_labels(self, path):
    with open(path, 'r') as f:
      return {i: line.strip() for i, line in enumerate(f.readlines())}

  def classify(self, original_image, top_k=1):
    start_time = time.time()
    image = cv2.resize(original_image, (self.height, self.width))
    image = self.preprocess(image)
    results = inf.do_inference(self.context, self.engine, image, self.h_input, self.d_input, self.h_output, self.d_output, self.stream, 1, self.height, self.width)
    elapsed_ms = (time.time() - start_time) * 1000
    FPS = 1000/elapsed_ms
    idx = np.argmax(results)
    prob = results[idx]
    text = 'Class: %s Confidence: %.2f  FPS: %.1f' % (self.labels[idx], prob, FPS)
    cv2.putText(original_image, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, self.width/400, (0, 0, 255), 2, True)

    return original_image


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--source', help='File path or camera', default=0)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
args = parser.parse_args()

classifier = Classifier(args.labels,args.model)
camera = cv2.VideoCapture(args.source)

while(camera.isOpened()):
    ret, frame = camera.read()
    image = classifier.classify(frame)
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
