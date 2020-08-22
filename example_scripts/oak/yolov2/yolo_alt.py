import consts.resource_paths
import cv2
import depthai
import argparse
import time 
import numpy as np
from box import BoundBox, nms_boxes, boxes_to_array, to_minmax, draw_boxes

class Detector(object):

    def __init__(self, label_file, model_file, threshold):
            
        self._threshold = float(threshold)
        self.labels = self.load_labels(label_file)

    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.read().replace('"','').split(','))}

    def parse(self, original_image, tensor):
        #start_time = time.time()
        #elapsed_ms = (time.time() - start_time) * 1000
        #fps  = 1 / elapsed_ms*1000
        #print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))
        boxes, probs = self.run(tensor)

        
        def _to_original_scale(boxes):
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= 224
            minmax_boxes[:,2] *= 224
            minmax_boxes[:,1] *= 224
            minmax_boxes[:,3] *= 224
            return minmax_boxes.astype(np.int)
        
        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            #print(boxes)
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

if __name__ == "__main__" :
    detector = Detector(args.labels, args.model, args.threshold)
    
    
    if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
        raise RuntimeError("Error initializing device. Try to reset it.")

    p = depthai.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": args.model,
        "blob_file_config": 'yolov2/YOLO_best_mAP_alt.json'
          }
        })

    if p is None:
        raise RuntimeError("Error initializing pipelne")
    recv = False
    while True:
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

        for nnet_packet in nnet_packets:
            raw_detections = nnet_packet.get_tensor(0)
            raw_detections.dtype = np.float16
            raw_detections = np.squeeze(raw_detections)
            output_shape = [5, 6, 7, 7]
            output = np.reshape(raw_detections, output_shape)
            output = np.transpose(output, (2, 3, 0, 1))
            recv = True
            
        for packet in data_packets:
            if packet.stream_name == 'previewout':
                data = packet.getData()
                data0 = data[0, :, :]
                data1 = data[1, :, :]
                data2 = data[2, :, :]
                frame = cv2.merge([data0, data1, data2])
                if recv:
                    frame = detector.parse(frame, output)
                cv2.imshow('previewout', frame)

        if cv2.waitKey(1) == ord('q'):
            break

del p
depthai.deinit_device()


