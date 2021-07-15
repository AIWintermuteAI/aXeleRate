# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
This file contains helper functions for reading video/image data and
 pre/postprocessing of video/image data using OpenCV.
"""

import os
import cv2
import numpy as np

def preprocess(img):

    img = img.astype(np.float32)
    img = img / 255.
    img = img - 0.5
    img = img * 2.
    img = img[:, :, ::-1]
    img = np.expand_dims(img, 0)
    return img

def decode_yolov2(netout, 
                  nms_threshold = 0.2,
                  threshold = 0.3, 
                  anchors = [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025]):

    #Convert Yolo network output to bounding box

    netout = netout[0].reshape(7,7,5,6)
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > threshold
    
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
    
    boxes = nms_boxes(boxes, len(classes), nms_threshold, threshold)

    if len(boxes) > 0:
        return boxes_to_array(boxes)
    else:
        return []

def decode_yolov3(netout, 
                  nms_threshold = 0.2,
                  threshold = 0.3, 
                  anchors = [[[0.76120044, 0.57155991], [0.6923348, 0.88535553], [0.47163042, 0.34163313]],
                                 [[0.33340788, 0.70065861], [0.18124964, 0.38986752], [0.08497349, 0.1527057 ]]]):

    #Convert Yolo network output to bounding box

    boxes = []

    for l, output in enumerate(netout):
        grid_h, grid_w, nb_box = output.shape[0:3]
        
        # decode the output by the network
        output[..., 4] = _sigmoid(output[..., 4])
        output[..., 5:] = output[..., 4][..., np.newaxis] * _sigmoid(output[..., 5:])
        output[..., 5:] *= output[..., 5:] > threshold
        
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
                        w = anchors[l][b][0] * np.exp(w) # unit: image width
                        h = anchors[l][b][1] * np.exp(h) # unit: image height
                        confidence = output[row, col, b, 4]
                        box = BoundBox(x, y, w, h, confidence, classes)
                        boxes.append(box)

    boxes = nms_boxes(boxes, len(classes), nms_threshold, threshold)

    if len(boxes) > 0:
        return boxes_to_array(boxes)
    else:
        return []

def decode_classifier(netout, top_k=3):
    netout = netout[0]
    ordered = np.argsort(netout)
    results = [(i, netout[i]) for i in ordered[-top_k:][::-1]]
    return results

def decode_segnet(netout, labels, class_colors):
    netout = netout[0] 

    seg_arr = netout.argmax(axis=2)

    seg_img = np.zeros((netout.shape[0], netout.shape[1], 3))

    for c in range(len(labels)):
        seg_img[:, :, 0] += ((seg_arr[:, :] == c)*(class_colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr[:, :] == c)*(class_colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr[:, :] == c)*(class_colors[c][2])).astype('uint8')

    return seg_img

def get_legends(class_names, colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25), 150, 3), dtype="uint8") + 255

    for (i, (class_name, color)) in enumerate(zip(class_names.values() , colors)):
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (125, (i * 25)), (150, (i * 25) + 25), tuple(color), -1)

    return legend 

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2 ).astype('uint8')
    return fused_img 

def concat_lenends(seg_img, legend_img):
    
    seg_img[:legend_img.shape[0],:legend_img.shape[1]] = np.copy(legend_img)

    return seg_img

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def resize_with_aspect_ratio(frame: np.ndarray, input_binding_info: tuple):
    """
    Resizes frame while maintaining aspect ratio, padding any empty space.

    Args:
        frame: Captured frame.
        input_binding_info: Contains shape of model input layer.

    Returns:
        Frame resized to the size of model input layer.
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]
    model_height, model_width = list(input_binding_info[1].GetShape())[1:3]

    if aspect_ratio >= 1.0:
        new_height, new_width = int(model_width / aspect_ratio), model_width
        b_padding, r_padding = model_height - new_height, 0
    else:
        new_height, new_width = model_height, int(model_height * aspect_ratio)
        b_padding, r_padding = 0, model_width - new_width

    # Resize and pad any empty space
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame = cv2.copyMakeBorder(frame, top=0, bottom=b_padding, left=0, right=r_padding,
                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return frame


def create_video_writer(video, video_path, output_name):
    """
    Creates a video writer object to write processed frames to file.

    Args:
        video: Video capture object, contains information about data source.
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video writer object.
    """
    _, ext = os.path.splitext(video_path)

    i, filename = 0, output_name + ext
    while os.path.exists(filename):
        i += 1
        filename = output_name + str(i) + ext

    video_writer = cv2.VideoWriter(filename=filename,
                                   fourcc=get_source_encoding_int(video),
                                   fps=int(video.get(cv2.CAP_PROP_FPS)),
                                   frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    return video_writer


def init_video_file_capture(video_path, output_name):
    """
    Creates a video capture object from a video file.

    Args:
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video capture object to capture frames, video writer object to write processed
        frames to file, plus total frame count of video source to iterate through.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found for: {video_path}')
    video = cv2.VideoCapture(video_path)
    if not video.isOpened:
        raise RuntimeError(f'Failed to open video capture from file: {video_path}')

    video_writer = create_video_writer(video, video_path, output_name)
    iter_frame_count = range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    return video, video_writer, iter_frame_count

def draw_bounding_boxes(frame, detections, labels=None, processing_function=None):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.

    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        labels: Dictionary of labels and colors keyed on the classification index.
    """
    def _to_original_scale(boxes, frame_height, frame_width):
        minmax_boxes = np.empty(shape=(4, ), dtype=np.int)

        cx = boxes[0] * frame_width
        cy = boxes[1] * frame_height
        w = boxes[2] * frame_width
        h = boxes[3] * frame_height
        
        minmax_boxes[0] = cx - w/2
        minmax_boxes[1] = cy - h/2
        minmax_boxes[2] = cx + w/2
        minmax_boxes[3] = cy + h/2

        return minmax_boxes

    color = (0, 255, 0)
    label_color = (125, 125, 125)

    for i in range(len(detections)):
        class_idx, box, confidence = [d for d in detections[i]]

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]

        x_min, y_min, x_max, y_max = _to_original_scale(box, frame_height, frame_width)
        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        if processing_function:
            roi_img = frame[y_min:y_max, x_min:x_max]
            label = processing_function(roi_img)
        else:
            # Create label for detected object class
            label = labels[class_idx].capitalize() 
            label = f'{label} {confidence * 100:.1f}%'

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.55 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.50, label_color, 1, cv2.LINE_AA)

def draw_classification(frame, classifications, labels):

    for i in range(len(classifications)):
        label_id, prob = classifications[i]
        text = '%s : %.2f' % (labels[label_id], prob)
        cv2.putText(frame, text, (10, 20*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, True)

def get_source_encoding_int(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FOURCC))

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

    def get_label(self):
        return np.argmax(self.classes)
    
    def get_score(self):
        return self.classes[self.get_label()]
    
    def iou(self, bound_box):
        b1 = self.as_centroid()
        b2 = bound_box.as_centroid()
        return centroid_box_iou(b1, b2)

    def as_centroid(self):
        return np.array([self.x, self.y, self.w, self.h])
    

def boxes_to_array(bound_boxes):
    """
    # Args
        boxes : list of BoundBox instances
    
    # Returns
        centroid_boxes : (N, 4)
        probs : (N, nb_classes)
    """
    temp_list = []
    for box in bound_boxes:
        temp_list.append([np.argmax(box.classes), np.asarray([box.x, box.y, box.w, box.h]), np.max(box.classes)])

    return np.array(temp_list)


def nms_boxes(boxes, n_classes, nms_threshold=0.3, obj_threshold=0.3):
    """
    # Args
        boxes : list of BoundBox
    
    # Returns
        boxes : list of BoundBox
            non maximum supressed BoundBox instances
    """
    # suppress non-maximal boxes
    for c in range(n_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    return boxes

def centroid_box_iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
    
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    _, _, w1, h1 = box1.reshape(-1,)
    _, _, w2, h2 = box2.reshape(-1,)
    x1_min, y1_min, x1_max, y1_max = to_minmax(box1.reshape(-1,4)).reshape(-1,)
    x2_min, y2_min, x2_max, y2_max = to_minmax(box2.reshape(-1,4)).reshape(-1,)
            
    intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union

def to_minmax(centroid_boxes):
    centroid_boxes = centroid_boxes.astype(np.float)
    minmax_boxes = np.zeros_like(centroid_boxes)
    
    cx = centroid_boxes[:,0]
    cy = centroid_boxes[:,1]
    w = centroid_boxes[:,2]
    h = centroid_boxes[:,3]
    
    minmax_boxes[:,0] = cx - w/2
    minmax_boxes[:,1] = cy - h/2
    minmax_boxes[:,2] = cx + w/2
    minmax_boxes[:,3] = cy + h/2
    return minmax_boxes