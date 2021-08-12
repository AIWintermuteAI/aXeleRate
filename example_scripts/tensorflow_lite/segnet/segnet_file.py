import time
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

import random
random.seed(0)

from cv_utils import init_video_file_capture, decode_segnet, get_legends, overlay_seg_image, concat_lenends, preprocess
from tflite_runtime.interpreter import Interpreter

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

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.file, 'segnet_demo')

    if not os.path.exists(args.labels[0]):
        labels = args.labels
    else:   
        labels = load_labels(args.labels[0])

    class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(256)]
    legend_img = get_legends(labels, class_colors)

    frame_num = len(frame_count)
    times = []

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        start_time = time.time()
        results = segmentation_network.run(frame)
        elapsed_ms = (time.time() - start_time) * 1000

        seg_img = decode_segnet(results, labels, class_colors)

        if args.overlay == True:
            seg_img = overlay_seg_image(frame, seg_img)

        frame = concat_lenends(seg_img, legend_img)

        times.append(elapsed_ms)
        video_writer.write(frame)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1 / time to process loop

if __name__ == "__main__" :

    print("OpenCV version: {}".format(cv2. __version__))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', nargs="+", help='File path of labels file.', required=True)
    parser.add_argument('--overlay', help='Overlay original image.', default=True)
    parser.add_argument('--file', help='File path of video file', default=None)
    args = parser.parse_args()

    segmentation_network = NetworkExecutor(args.model)

    main(args)
    
