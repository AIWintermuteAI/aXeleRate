from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter

from flask import Flask, render_template, request, Response
from camera_opencv import Camera

app = Flask (__name__, static_url_path = '')

class Classifier:

  def __init__(self,label_file,model_file):
    self.labels = self.load_labels(label_file)
    self.interpreter = Interpreter(model_file)
    self.interpreter.allocate_tensors()
    _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']

  def load_labels(self, path):
    with open(path, 'r') as f:
      return {i: line.strip() for i, line in enumerate(f.readlines())}

  def set_input_tensor(self, image):
    tensor_index = self.interpreter.get_input_details()[0]['index']
    input_tensor = self.interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def classify(self, original_image, top_k=1):
    start_time = time.time()
    image = cv2.resize(original_image, (self.height, self.width))

    """Returns a sorted array of classification results."""
    self.set_input_tensor(image)
    self.interpreter.invoke()
    output_details = self.interpreter.get_output_details()[0]
    output = np.squeeze(self.interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
      scale, zero_point = output_details['quantization']
      output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    results =  [(i, output[i]) for i in ordered[:top_k]]

    elapsed_ms = (time.time() - start_time) * 1000
    label_id, prob = results[0]
    text = 'Class: %s Confidence: %.2f  TIME: %.1fms' % (self.labels[label_id], prob, elapsed_ms)
    cv2.putText(original_image, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, self.width/400, (0, 0, 255), 2, True)

    return cv2.imencode('.jpg', original_image)[1].tobytes()

@app.route ("/")
def index ( ):
   return render_template ('index.html', name = None)

def gen(camera):
    while True:
        frame = camera.get_frame()
        image = classifier.classify(frame)
        yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
args = parser.parse_args()

classifier = Classifier(args.labels,args.model)

if __name__ == "__main__" :
   app.run (host = '0.0.0.0', port = 5000, debug = True)
