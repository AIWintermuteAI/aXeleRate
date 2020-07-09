from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re
import time
import argparse
import cv2
import numpy as np
import random
import tensorflow as tf

from flask import Flask, render_template, request, Response

app = Flask (__name__, static_url_path = '')

random.seed(0)

class Segnet(object):
    def __init__(self, model_file, label_file, overlay):
        self.interpreter = tf.lite.Interpreter(model_file)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']
        self.labels = self.load_labels(label_file)
        self.class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]
        self.legend_img = self.get_legends(self.labels)
        self.overlay = overlay

    def load_labels(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.read().replace('"','').split(',')]

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32)
        img = img / 255.
        img = img - 0.5
        img = img * 2.
        img = img[:, :, ::-1]
        img = np.expand_dims(img, 0)
        return img

    def get_legends(self, class_names):
        colors=self.class_colors
        n_classes = len(class_names)
        legend = np.zeros(((len(class_names) * 25) + 25, 125, 3), dtype="uint8") + 255

        for (i, (class_name, color)) in enumerate(zip(class_names , colors)):
            color = [int(c) for c in color]
            cv2.putText(legend, class_name, (5, (i * 25) + 17),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),tuple(color), -1)
            
        return legend 

    def overlay_seg_image(self, inp_img , seg_img):
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

        fused_img = (inp_img/2 + seg_img/2 ).astype('uint8')
        return fused_img 

    def concat_lenends(self, seg_img , legend_img):
        
        new_h = np.maximum( seg_img.shape[0] , legend_img.shape[0] )
        new_w = seg_img.shape[1] + legend_img.shape[1]
        out_img = np.zeros((new_h ,new_w , 3  )).astype('uint8') + legend_img[0 , 0 , 0 ]
        out_img[ :legend_img.shape[0] , :  legend_img.shape[1] ] = np.copy(legend_img)
        out_img[ :seg_img.shape[0] , legend_img.shape[1]: ] = np.copy(seg_img)

        return out_img

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def segment_objects(self, image):
        img = self.preprocess(image)
        self.interpreter.set_tensor(self.tensor_index, img)
        self.interpreter.invoke()
        seg_arr = self.get_output_tensor(0)
        return seg_arr

    def segment(self, original_image):
        output_height, output_width = original_image.shape[0:2]

        start_time = time.time()
        results = self.segment_objects(original_image)
        elapsed_ms = (time.time() - start_time) * 1000
        fps  = 1 / elapsed_ms*1000
        print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))

        seg_arr = results.argmax(axis=2)

        seg_img = np.zeros((results.shape[0], results.shape[1], 3))

        for c in range(len(self.labels)):
            seg_img[:, :, 0] += ((seg_arr[:, :] == c)*(self.class_colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg_arr[:, :] == c)*(self.class_colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg_arr[:, :] == c)*(self.class_colors[c][2])).astype('uint8')

        seg_img = cv2.resize(seg_img, (output_width, output_height))
        if self.overlay == True:
            seg_img = self.overlay_seg_image(original_image, seg_img)
        seg_img = self.concat_lenends(seg_img, self.legend_img)

        return cv2.imencode('.jpg', seg_img)[1].tobytes()


@app.route ("/")
def index ( ):
   return render_template ('index.html', name = None)

def gen(camera):
    while True:
        frame = camera.get_frame()
        image = segnet.segment(frame)
        yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--labels', help='File path of labels file.', required=True)
parser.add_argument('--overlay', help='Overlay original image.', default=True)
parser.add_argument('--source', help='picamera or cv', default='cv')
args = parser.parse_args()

if args.source == "cv":
    from camera_opencv import Camera
elif args.source == "picamera":
    from camera_pi import Camera

segnet = Segnet(args.model, args.labels, args.overlay)

if __name__ == "__main__" :
   app.run (host = '0.0.0.0', port = 5000, debug = True)
    
