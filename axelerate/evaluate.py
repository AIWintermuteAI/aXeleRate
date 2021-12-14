import os
import argparse
import json
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import backend as K 

from axelerate.networks.yolo.frontend import create_yolo
from axelerate.networks.yolo.backend.utils.box import draw_boxes
from axelerate.networks.yolo.backend.utils.annotation import parse_annotation
from axelerate.networks.yolo.backend.utils.eval.fscore import count_true_positives, calc_score
from axelerate.networks.segnet.frontend_segnet import create_segnet
from axelerate.networks.classifier.frontend_classifier import get_labels, create_classifier

K.clear_session()

DEFAULT_THRESHOLD = 0.3

def save_report(config, report, report_file):
    with open(report_file, 'w') as outfile:
        outfile.write("REPORT\n")
        outfile.write(str(report))
        outfile.write("\nCONFIG\n")
        outfile.write(json.dumps(config, indent=4, sort_keys=False))

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
    input_image = cv2.resize(input_image, (network.input_size[1], network.input_size[0]))
    input_image = network.norm(input_image)
    input_image = np.expand_dims(input_image, 0)
    return orig_image, input_image

def setup_evaluation(config, weights, threshold = None):
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
    dirname = os.path.dirname(weights)

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

        # 2. Load the pretrained weights
        classifier.load_weights(weights)

        report, cm = classifier.evaluate(config['train']['valid_image_folder'], 16)
        save_report(config, report, os.path.join(dirname, 'report.txt'))

    if config['model']['type']=='SegNet':
        print('Segmentation')           
        # 1. Construct the model 
        segnet = create_segnet(config['model']['architecture'],
                                   input_size,
                                   config['model']['n_classes'])   
        # 2. Load the pretrained weights (if any) 
        segnet.load_weights(weights)
        report = segnet.evaluate(config['train']['valid_image_folder'], config['train']['valid_annot_folder'], 2)
        save_report(config, report, os.path.join(dirname, 'report.txt'))
        print(report)

    if config['model']['type']=='Detector':
        # 2. create yolo instance & predict
        yolo = create_yolo(config['model']['architecture'],
                           config['model']['labels'],
                           input_size,
                           config['model']['anchors'],
                           config['model']['obj_thresh'],
                           config['model']['iou_thresh'],
                           config['model']['coord_scale'],
                           config['model']['object_scale'],
                           config['model']['no_object_scale'],                           
                           config['weights']['backend'])    
        yolo.load_weights(weights)

        # 3. read image
        annotations = parse_annotation(config['train']['valid_annot_folder'],
                                       config['train']['valid_image_folder'],
                                       config['model']['labels'],
                                       is_only_detect=config['train']['is_only_detect'])

        threshold = threshold if threshold else config['model']['obj_thresh']

        dirname = os.path.join(os.path.dirname(weights), 'Inference_results') #temporary

        if os.path.isdir(dirname):
            print("Folder {} is already exists. Image files in directory might be overwritten".format(dirname))
        else:
            print("Folder {} is created.".format(dirname))
            os.makedirs(dirname)

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
            prediction_time, boxes, scores = yolo.predict(input_image, height, width, float(threshold))
            classes = np.argmax(scores, axis=1) if len(scores) > 0 else []
            inference_time.append(prediction_time)

            # 4. save detection result
            orig_image = draw_boxes(orig_image, boxes, scores, classes, config['model']['labels'])
            output_path = os.path.join(dirname, os.path.split(img_fname)[-1])
            cv2.imwrite(output_path, orig_image)
            print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))
            n_true_positives += count_true_positives(boxes, true_boxes, classes, true_labels)
            n_truth += len(true_boxes)
            n_pred += len(boxes)

        report = calc_score(n_true_positives, n_truth, n_pred)
        save_report(config, report, os.path.join(dirname, 'report.txt'))
        print(report)

        if len(inference_time)>1:
            print("Average prediction time:{} ms".format(sum(inference_time[1:])/len(inference_time[1:])))

if __name__ == '__main__':
    # 1. extract arguments

    argparser = argparse.ArgumentParser(
        description='Run evaluation script')

    argparser.add_argument(
        '-c',
        '--config',
        help='path to configuration file')

    argparser.add_argument(
        '-t',
        '--threshold',
        help='detection threshold')

    argparser.add_argument(
        '-w',
        '--weights',
        help='trained weight files')

    args = argparser.parse_args()
    with open(args.config) as config_buffer:
        config = json.loads(config_buffer.read())
    setup_evaluation(config, args.weights, args.threshold)
