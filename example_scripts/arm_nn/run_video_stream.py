"""
Object detection demo that takes a video stream from a device, runs inference
on each frame producing bounding boxes and labels around detected objects,
and displays a window with the latest processed frame.
"""
import os
import sys
import time
script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from yolov2 import yolo_processing, yolo_resize_factor

from cv_utils import init_video_stream_capture, resize_with_aspect_ratio
from network_executor import ArmnnNetworkExecutor
import pyarmnn as ann


def preprocess(frame: np.ndarray, input_binding_info: tuple):
    """
    Takes a frame, resizes, swaps channels and converts data type to match
    model input layer. The converted frame is wrapped in a const tensor
    and bound to the input tensor.

    Args:
        frame: Captured frame from video.
        input_binding_info:  Contains shape and data type of model input layer.

    Returns:
        Input tensor.
    """
    # Swap channels and resize frame to model resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = resize_with_aspect_ratio(frame, input_binding_info)

    # Expand dimensions and convert data type to match model input
    data_type = np.float32 if input_binding_info[1].GetDataType() == ann.DataType_Float32 else np.uint8
    resized_frame = np.expand_dims(np.asarray(resized_frame, dtype=data_type), axis=0)
    resized_frame /= 255.
    resized_frame -= 0.5
    resized_frame *= 2
    assert resized_frame.shape == tuple(input_binding_info[1].GetShape())

    input_tensors = ann.make_input_tensors([input_binding_info], [resized_frame])
    return input_tensors

def process_faces(frame, detections, executor_kp, resize_factor):
    kpts_list = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for detection in detections:
        box = detection[1].copy()
        for i in range(len(box)):
            box[i] = int(box[i] * resize_factor)
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

        face_img = frame[box[1]:box[3], box[0]:box[2]]

        face_img = cv2.resize(face_img, (128, 128)) 
        #cv2.imshow('PyArmNN Object Detection Demo face', face_img)
        face_img = face_img.astype(np.float32)
        face_img /= 127.5
        face_img -= 1.

        input_tensors = ann.make_input_tensors([executor_kp.input_binding_info], [face_img])

        plist = executor_kp.run(input_tensors)[0][0]

        le = (x + int(plist[0] * w+5), y + int(plist[1] * h+5))
        re = (x + int(plist[2] * w), y + int(plist[3] * h+5))
        n = (x + int(plist[4] * w), y + int(plist[5] * h))
        lm = (x + int(plist[6] * w), y + int(plist[7] * h))
        rm = (x + int(plist[8] * w), y + int(plist[9] * h))
        kpts = [le, re, n, lm, rm]

        kpts_list.append(kpts)

    return kpts_list

def draw_bounding_boxes(frame: np.ndarray, detections: list, resize_factor, kpts):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.

    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        labels: Dictionary of labels and colors keyed on the classification index.
    """
    for detection in detections:
        class_idx, box, confidence = [d for d in detection]
        label, color = 'Person', (0, 255, 0)

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = [int(position * resize_factor) for position in box]

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label for detected object class
        label = f'{label} {confidence * 100:.1f}%'
        label_color = (0, 0, 0) if sum(color)>200 else (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.55 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.50,
                    label_color, 1, cv2.LINE_AA)

        for kpt_set in kpts:
            for kpt in kpt_set:

                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (255, 0, 0), 2)
def main(args):
    video = init_video_stream_capture(args.video_source)

    executor_fd = ArmnnNetworkExecutor(args.fd_model_file_path, args.preferred_backends)
    executor_kp = ArmnnNetworkExecutor(args.kp_model_file_path, args.preferred_backends)    

    process_output, resize_factor = yolo_processing, yolo_resize_factor(video, executor_fd.input_binding_info)

    while True:

        frame_present, frame = video.read()
        frame = cv2.flip(frame, 1)  # Horizontally flip the frame
        if not frame_present:
            raise RuntimeError('Error reading frame from video stream')
        input_tensors = preprocess(frame, executor_fd.input_binding_info)
        print("Running inference...")

        start_time = time.time() 
        output_result = executor_fd.run(input_tensors)
        detections = process_output(output_result)
        kpts = process_faces(frame, detections, executor_kp, resize_factor)

        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
        print("Time(ms): ", (time.time() - start_time)*1000) 

        draw_bounding_boxes(frame, detections, resize_factor, kpts)
        cv2.imshow('PyArmNN Object Detection Demo', frame)

        if cv2.waitKey(1) == 27:
            print('\nExit key activated. Closing video...')
            break
    video.release(), cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_source', type=int, default=0,
                        help='Device index to access video stream. Defaults to primary device camera at index 0')

    parser.add_argument('--fd_model_file_path', required=True, type=str,
                        help='Path to the Object Detection model to use')
    parser.add_argument('--kp_model_file_path', required=True, type=str,
                        help='Path to the Object Detection model to use')

    parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                        help='Takes the preferred backends in preference order, separated by whitespace, '
                             'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                             'Defaults to [CpuAcc, CpuRef]')
    args = parser.parse_args()
    main(args)
