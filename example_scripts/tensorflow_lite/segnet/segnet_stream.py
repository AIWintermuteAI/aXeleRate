import time
import argparse
import os
import cv2
import numpy as np

import random
random.seed(0)

from cv_utils import decode_segnet, get_legends, overlay_seg_image, concat_lenends, preprocess

from tflite_runtime.interpreter import Interpreter
from flask import Flask, render_template, request, Response

app = Flask (__name__, static_url_path = '')

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.read().replace('"','').split(','))}

class NetworkExecutor(object):

    def __init__(self, model_file):

        self.interpreter = Interpreter(model_file, num_threads=3)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']

    def get_output_tensors(self):

      output_details = self.interpreter.get_output_details()
      tensor_indices = []
      tensor_list = []

      for output in output_details:
            tensor = np.squeeze(self.interpreter.get_tensor(output['index']))
            tensor_list.append(tensor)

      return tensor_list

    def run(self, image):
        if image.shape[1:2] != (self.input_height, self.input_width):
            img = cv2.resize(image, (self.input_width, self.input_height))
        img = preprocess(img)
        self.interpreter.set_tensor(self.tensor_index, img)
        self.interpreter.invoke()
        return self.get_output_tensors()

class Segnet(NetworkExecutor):

    def __init__(self, label_file, model_file, overlay):
        super().__init__(model_file)

        if not os.path.exists(label_file):
            self.labels = [label_file]
        else:   
            self.labels = load_labels(label_file)

        self.class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(256)]
        self.legend_img = get_legends(self.labels, self.class_colors)
        self.overlay = overlay 

    def segment(self, frame):
        start_time = time.time()
        results = self.run(frame)
        elapsed_ms = (time.time() - start_time) * 1000

        seg_img = decode_segnet(results, self.labels, self.class_colors)

        if args.overlay == True:
            seg_img = overlay_seg_image(frame, seg_img)

        frame = concat_lenends(seg_img, self.legend_img)

        fps  = 1 / elapsed_ms*1000
        print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))

        return cv2.imencode('.jpg', frame)[1].tobytes()

@app.route("/")
def index():
   return render_template('index.html', name = None)

def gen(camera):
    while True:
        frame = camera.get_frame()
        image = segnet.segment(frame)
        yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
parser.add_argument('--overlay', help='Overlay original image.', default=True)
parser.add_argument('--source', help='picamera or cv', default='cv')
args = parser.parse_args()

if args.source == "cv":
    from camera_opencv import Camera
    source = 0
elif args.source == "picamera":
    from camera_pi import Camera
    source = 0
    
Camera.set_video_source(source)

segnet = Segnet(args.labels, args.model, args.overlay)

if __name__ == "__main__" :
   app.run(host = '0.0.0.0', port = 5000, debug = True)
    
