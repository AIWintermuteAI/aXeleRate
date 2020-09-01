# -*- coding: utf-8 -*-

import argparse
import json
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import backend as K 
from axelerate.networks.yolo.frontend import create_yolo
from axelerate.networks.yolo.backend.utils.box import draw_scaled_boxes
from axelerate.networks.yolo.backend.utils.annotation import parse_annotation
from axelerate.networks.yolo.backend.utils.eval.fscore import count_true_positives, calc_score
from axelerate.networks.segnet.frontend_segnet import create_segnet
from axelerate.networks.segnet.predict import predict
from axelerate.networks.classifier.frontend_classifier import get_labels,create_classifier

import os
import glob
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

K.clear_session()

DEFAULT_THRESHOLD = 0.3

argparser = argparse.ArgumentParser(
    description='Run inference script')

argparser.add_argument(
    '-c',
    '--config',
    help='path to configuration file')

argparser.add_argument(
    '-t',
    '--threshold',
    default=DEFAULT_THRESHOLD,
    help='detection threshold')

argparser.add_argument(
    '-w',
    '--weights',
    help='trained weight files')

def show_image(filename):
    image = mpimg.imread(filename)
    plt.figure()
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(filename)

def prepare_image(img_path, network):
    orig_image = cv2.imread(img_path)
    input_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
    input_image = cv2.resize(input_image, (network._input_size[1],network._input_size[0]))
    input_image = network._norm(input_image)
    input_image = np.expand_dims(input_image, 0)
    return orig_image, input_image

def setup_inference(config, weights,threshold=0.3, path=None):
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    #added for compatibility with < 0.5.7 versions
    try:
        input_size = config['model']['input_size'][:]
    except:
        input_size = [config['model']['input_size'],config['model']['input_size']]

    """make directory to save inference results """
    dirname = os.path.join(os.path.dirname(weights),'Inference_results')
    if os.path.isdir(dirname):
        print("Folder {} is already exists. Image files in directory might be overwritten".format(dirname))
    else:
        print("Folder {} is created.".format(dirname))
        os.makedirs(dirname)

    if config['model']['type']=='SegNet':
        print('Segmentation')           
        # 1. Construct the model 
        segnet = create_segnet(config['model']['architecture'],
                                   input_size,
                                   config['model']['n_classes'])   
        # 2. Load the pretrained weights (if any) 
        segnet.load_weights(weights)
        for filename in os.listdir(config['train']['valid_image_folder']):
            filepath = os.path.join(config['train']['valid_image_folder'],filename)
            orig_image, input_arr = prepare_image(filepath, segnet)
            out_fname = os.path.join(dirname, os.path.basename(filename))
            predict(model=segnet._network, inp=input_arr, image = orig_image, out_fname=out_fname)

    if config['model']['type']=='Classifier':
        print('Classifier')    
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_labels(config['train']['train_image_folder'])
        # 1.Construct the model 
        classifier = create_classifier(config['model']['architecture'],
                                       labels,
                                       input_size,
                                       config['model']['fully-connected'],
                                       config['model']['dropout'])   
        # 2. Load the pretrained weights (if any) 
        classifier.load_weights(weights)
        font = cv2.FONT_HERSHEY_SIMPLEX
        valid_image_folder = config['train']['valid_image_folder']
        image_files_list = glob.glob(valid_image_folder + '/**/*.jpg', recursive=True)
        inference_time = []
        for filename in image_files_list:
            output_path = os.path.join(dirname, os.path.basename(filename))
            orig_image, input_image = prepare_image(filename, classifier)
            prediction_time, img_class, prob = classifier.predict(input_image)
            inference_time.append(prediction_time)
            
            # label shape and colorization
            text = "{}:{:.2f}".format(img_class[0], prob[0])
            background_color = (70, 120, 70) # grayish green background for text
            text_color = (255, 255, 255)   # white text

            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            left = 10
            top = 30 - size[1]
            right = left + size[0]
            bottom = top + size[1]

            # set up the colored rectangle background for text
            cv2.rectangle(orig_image, (left - 1, top - 5),(right + 1, bottom + 1), background_color, -1)
            # set up text
            cv2.putText(orig_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.imwrite(output_path, orig_image)
            show_image(output_path)
            print("{}:{}".format(img_class[0], prob[0]))
        if len(inference_time)>1:
            print("Average prediction time:{} ms".format(sum(inference_time[1:])/len(inference_time[1:])))

    if config['model']['type']=='Detector':
        # 2. create yolo instance & predict
        yolo = create_yolo(config['model']['architecture'],
                           config['model']['labels'],
                           input_size,
                           config['model']['anchors'])
        yolo.load_weights(weights)

        # 3. read image
        annotations = parse_annotation(config['train']['valid_annot_folder'],
                                       config['train']['valid_image_folder'],
                                       config['model']['labels'],
                                       is_only_detect=config['train']['is_only_detect'])

        n_true_positives = 0
        n_truth = 0
        n_pred = 0
        inference_time = []
        for i in range(len(annotations)):
            img_path = annotations.fname(i)
            img_fname = os.path.basename(img_path)
            true_boxes = annotations.boxes(i)
            true_labels = annotations.code_labels(i)

            orig_image, input_image = prepare_image(img_path, yolo)
            height, width = orig_image.shape[:2]
            prediction_time, boxes, probs = yolo.predict(input_image, height, width, float(threshold))
            inference_time.append(prediction_time)
            labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
            # 4. save detection result
            orig_image = draw_scaled_boxes(orig_image, boxes, probs, config['model']['labels'])
            output_path = os.path.join(dirname, os.path.split(img_fname)[-1])
            cv2.imwrite(output_path, orig_image)
            print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))
            show_image(output_path)
            n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
            n_truth += len(true_boxes)
            n_pred += len(boxes)
        print(calc_score(n_true_positives, n_truth, n_pred))
        if len(inference_time)>1:
            print("Average prediction time:{} ms".format(sum(inference_time[1:])/len(inference_time[1:])))

if __name__ == '__main__':
    # 1. extract arguments

    argparser = argparse.ArgumentParser(
        description='Run inference script')

    argparser.add_argument(
        '-c',
        '--config',
        help='path to configuration file')

    argparser.add_argument(
        '-t',
        '--threshold',
        default=0.3,
        help='detection threshold')

    argparser.add_argument(
        '-w',
        '--weights',
        help='trained weight files')

    args = argparser.parse_args()
    with open(args.config) as config_buffer:
        config = json.loads(config_buffer.read())
    setup_inference(config, args.weights, args.threshold)
