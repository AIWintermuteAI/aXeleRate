import consts.resource_paths
import cv2
import depthai
import argparse
import time 
import numpy as np

IOU_THRESHOLD = 0.1
labels = ['null', 'kangaroo']
GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

def sigmoid(x):
    return 1.0 / (1 + np.exp(x * -1.0))


def calculate_overlap(x1, w1, x2, w2):
    box1_coordinate = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    box2_coordinate = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    overlap = box2_coordinate - box1_coordinate
    return overlap


def calculate_iou(box, truth):
    # calculate the iou intersection over union by first calculating the overlapping height and width
    width_overlap = calculate_overlap(box[0], box[2], truth[0], truth[2])
    height_overlap = calculate_overlap(box[1], box[3], truth[1], truth[3])
    # no overlap
    if width_overlap < 0 or height_overlap < 0:
        return 0

    intersection_area = width_overlap * height_overlap
    union_area = box[2] * box[3] + truth[2] * truth[3] - intersection_area
    iou = intersection_area / union_area
    return iou


def apply_nms(boxes):
    # sort the boxes by score in descending order
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    high_iou_objs = dict()
    # compare the iou for each of the detected objects
    for current_object in range(len(sorted_boxes)):
        if current_object in high_iou_objs:
            continue

        truth = sorted_boxes[current_object]
        for next_object in range(current_object + 1, len(sorted_boxes)):
            if next_object in high_iou_objs:
                continue
            box = sorted_boxes[next_object]
            iou = calculate_iou(box, truth)
            if iou >= IOU_THRESHOLD:
                high_iou_objs[next_object] = 1

    # filter and sort detected items
    filtered_result = list()
    for current_object in range(len(sorted_boxes)):
        if current_object not in high_iou_objs:
            filtered_result.append(sorted_boxes[current_object])
    return filtered_result

def post_processing(output, label_list, threshold):

    num_classes = 1
    num_grids = 7
    num_anchor_boxes = 5
    original_results = output.astype(np.float32)   

    # Tiny Yolo V2 uses a 13 x 13 grid with 5 anchor boxes for each grid cell.
    # This specific model was trained with the VOC Pascal data set and is comprised of 20 classes

    original_results = np.reshape(original_results, (num_anchor_boxes, 5+num_classes, num_grids, num_grids))
    reordered_results = np.transpose(original_results, (2, 3, 0, 1))
    reordered_results = np.reshape(reordered_results, (num_grids*num_grids, num_anchor_boxes, 5+num_classes))

    # The 125 results need to be re-organized into 5 chunks of 25 values
    # 20 classes + 1 score + 4 coordinates = 25 values
    # 25 values for each of the 5 anchor bounding boxes = 125 values
    #reordered_results = np.zeros((13 * 13, 5, 25))

    index = 0
    #for row in range( num_grids ):
    #    for col in range( num_grids ):
    #        for b_box_voltron in range(125):
    #            b_box = row * num_grids + col
    #            b_box_num = int(b_box_voltron / 25)
    #            b_box_info = b_box_voltron % 25
    #            reordered_results[b_box][b_box_num][b_box_info] = original_results[row][col][b_box_voltron]

    # shapes for the 5 Tiny Yolo v2 bounding boxes
    anchor_boxes = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    boxes = list()
    # iterate through the grids and anchor boxes and filter out all scores which do not exceed the DETECTION_THRESHOLD
    for row in range(num_grids):
        for col in range(num_grids):
            for anchor_box_num in range(num_anchor_boxes):
                box = list()
                class_list = list()
                current_score_total = 0
                # calculate the coordinates for the current anchor box
                box_x = (col + sigmoid(reordered_results[row * num_grids + col][anchor_box_num][0])) / 7.0
                box_y = (row + sigmoid(reordered_results[row * num_grids + col][anchor_box_num][1])) / 7.0
                box_w = (np.exp(reordered_results[row * num_grids + col][anchor_box_num][2]) *
                         anchor_boxes[2 * anchor_box_num]) / 7.0
                box_h = (np.exp(reordered_results[row * num_grids + col][anchor_box_num][3]) *
                         anchor_boxes[2 * anchor_box_num + 1]) / 7.0
                
                # find the class with the highest score
                for class_enum in range(num_classes):
                    class_list.append(reordered_results[row * num_grids + col][anchor_box_num][5 + class_enum])

                current_score_total = sum(class_list)
                for current_class in range(len(class_list)):
                    class_list[current_class] = class_list[current_class] * 1.0 / current_score_total

                # probability that the current anchor box contains an item
                object_confidence = sigmoid(reordered_results[row * num_grids + col][anchor_box_num][4])
                # highest class score detected for the object in the current anchor box
                highest_class_score = max(class_list)
                # index of the class with the highest score
                class_w_highest_score = class_list.index(max(class_list)) + 1
                # the final score for the detected object
                final_object_score = object_confidence * highest_class_score

                box.append(box_x)
                box.append(box_y)
                box.append(box_w)
                box.append(box_h)
                box.append(class_w_highest_score)
                box.append(object_confidence)
                box.append(highest_class_score)
                box.append(final_object_score)

                # filter out all detected objects with a score less than the threshold
                if final_object_score > threshold:
                    boxes.append(box)

    # gets rid of all duplicate boxes using non-maximal suppression
    results = apply_nms(boxes)
    return results

def show_tiny_yolo(results, original_img, is_depth=0):

    image_width = original_img.shape[1]
    image_height = original_img.shape[0]

    label_list = labels

    # calculate the actual box coordinates in relation to the input image
    print('\n Found this many objects in the image: ' + str(len(results)))
    for box in results:
        box_xmin = int((box[0] - box[2] / 2.0) * image_width)
        box_xmax = int((box[0] + box[2] / 2.0) * image_width)
        box_ymin = int((box[1] - box[3] / 2.0) * image_height)
        box_ymax = int((box[1] + box[3] / 2.0) * image_height)
        # ensure the box is not drawn out of the window resolution
        if box_xmin < 0:
            box_xmin = 0
        if box_xmax > image_width:
            box_xmax = image_width
        if box_ymin < 0:
            box_ymin = 0
        if box_ymax > image_height:
            box_ymax = image_height

        print(" - object: " + YELLOW + label_list[box[4]] + NOCOLOR + " is at left: " + str(box_xmin) + " top: " + str(box_ymin) + " right: " + str(box_xmax) + " bottom: " + str(box_ymax))

        # label shape and colorization
        label_text = label_list[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
        label_background_color = (70, 120, 70) # grayish green background for text
        label_text_color = (255, 255, 255)   # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(box_xmin)
        label_top = int(box_ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]

        # set up the colored rectangle background for text
        cv2.rectangle(original_img, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1),
                      label_background_color, -1)
        # set up text
        cv2.putText(original_img, label_text, (int(box_xmin), int(box_ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_text_color, 1)
        # set up the rectangle around the object
        cv2.rectangle(original_img, (int(box_xmin), int(box_ymin)), (int(box_xmax), int(box_ymax)), (0, 255, 0), 2)

    return original_img


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
parser.add_argument('--config', help='File path of config file.', required=True)
parser.add_argument('--threshold', help='Confidence threshold.', default=0.4)
args = parser.parse_args()

if __name__ == "__main__" :
    
    if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
        raise RuntimeError("Error initializing device. Try to reset it.")

    p = depthai.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": args.model,
        "blob_file_config": 'YOLO_best_mAP.json'
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
            recv = True
            
        for packet in data_packets:
            if packet.stream_name == 'previewout':
                data = packet.getData()
                data0 = data[0, :, :]
                data1 = data[1, :, :]
                data2 = data[2, :, :]
                frame = cv2.merge([data0, data1, data2])
                if recv:
                    filtered_objects = post_processing(raw_detections, ['kangaroo'], args.threshold)
                    frame = show_tiny_yolo(filtered_objects, frame, 0)
                cv2.imshow('previewout', frame)

        if cv2.waitKey(1) == ord('q'):
            break

del p
depthai.deinit_device()

