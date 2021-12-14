import glob
import os
import argparse
import json
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import backend as K 

from tensorflow.keras import backend as K 
from axelerate.networks.yolo.frontend import create_yolo
from axelerate.networks.yolo.backend.utils.box import draw_boxes
from axelerate.networks.segnet.frontend_segnet import create_segnet
from axelerate.networks.segnet.predict import visualize_segmentation
from axelerate.networks.classifier.frontend_classifier import get_labels, create_classifier

K.clear_session()
    
def show_image(filename):
    image = mpimg.imread(filename)
    plt.figure()
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(filename)

def prepare_image(img_path, network, input_size):
    orig_image = cv2.imread(img_path)
    input_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
    input_image = cv2.resize(input_image, (input_size[1], input_size[0]))
    input_image = network.norm(input_image)
    input_image = np.expand_dims(input_image, 0)
    return orig_image, input_image

def find_imgs(folder):
    ext_list = ['/**/*.jpg', '/**/*.jpeg', '/**/*.png', '/**/*.JPG', '/**/*.JPEG']
    image_files_list = []
    image_search = lambda ext : glob.glob(folder + ext, recursive=True)
    for ext in ext_list: image_files_list.extend(image_search(ext))
    return image_files_list

def setup_inference(config, weights, threshold = None, folder = None):
    try:
        matplotlib.use('TkAgg')
    except:
        pass

    #added for compatibility with < 0.5.7 versions
    try:
        input_size = config['model']['input_size'][:]
    except:
        input_size = [config['model']['input_size'], config['model']['input_size']]

    """make directory to save inference results """
    dirname = os.path.join(os.path.dirname(weights), 'Inference_results')
    if os.path.isdir(dirname):
        print("Folder {} is already exists. Image files in directory might be overwritten".format(dirname))
    else:
        print("Folder {} is created.".format(dirname))
        os.makedirs(dirname)

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
                                        
        # 2. Load the trained weights
        classifier.load_weights(weights)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        background_color = (70, 120, 70) # grayish green background for text
        text_color = (255, 255, 255)   # white text

        file_folder = folder if folder else config['train']['valid_image_folder']

        image_files_list = find_imgs(file_folder)
        
        inference_time = []
        for filepath in image_files_list:
            output_path = os.path.join(dirname, os.path.basename(filepath))
            orig_image, input_image = prepare_image(filepath, classifier, input_size)
            prediction_time, prob, img_class = classifier.predict(input_image)
            inference_time.append(prediction_time)
            
            text = "{}:{:.2f}".format(img_class, prob)

            # label shape and colorization
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            left = 10
            top = 35 - size[1]
            right = left + size[0]
            bottom = top + size[1]

            # set up the colored rectangle background for text
            cv2.rectangle(orig_image, (left - 1, top - 5),(right + 1, bottom + 1), background_color, -1)
            # set up text
            cv2.putText(orig_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.imwrite(output_path, orig_image)
            show_image(output_path)
            print("{}:{}".format(img_class, prob))

        if len(inference_time)>1:
            print("Average prediction time:{} ms".format(sum(inference_time[1:])/len(inference_time[1:])))

    if config['model']['type']=='SegNet':
        print('Segmentation')           
        # 1. Construct the model 
        segnet = create_segnet(config['model']['architecture'],
                                   input_size,
                                   config['model']['n_classes'])   
        # 2. Load the trained weights
        segnet.load_weights(weights)

        file_folder = folder if folder else config['train']['valid_image_folder']
        image_files_list = find_imgs(file_folder)

        inference_time = []
        for filepath in image_files_list:

            orig_image, input_image = prepare_image(filepath, segnet, input_size)
            out_fname = os.path.join(dirname, os.path.basename(filepath))
            prediction_time, output_array = segnet.predict(input_image)
            seg_img = visualize_segmentation(output_array, orig_image, segnet.n_classes, overlay_img = True)
            cv2.imwrite(out_fname, seg_img)
            show_image(out_fname)

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
        
        file_folder = folder if folder else config['train']['valid_image_folder']
        threshold = threshold if threshold else config['model']['obj_thresh']
        image_files_list = find_imgs(file_folder)

        inference_time = []
        for filepath in image_files_list:

            img_fname = os.path.basename(filepath)
            orig_image, input_image = prepare_image(filepath, yolo, input_size)
            height, width = orig_image.shape[:2]

            prediction_time, boxes, scores = yolo.predict(input_image, height, width, float(threshold))
            classes = np.argmax(scores, axis=1) if len(scores) > 0 else []
            print(classes)
            inference_time.append(prediction_time)

            # 4. save detection result
            orig_image = draw_boxes(orig_image, boxes, scores, classes, config['model']['labels'])
            output_path = os.path.join(dirname, os.path.basename(filepath))
            cv2.imwrite(output_path, orig_image)
            print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))
            show_image(output_path)

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
        help='detection threshold')

    argparser.add_argument(
        '-w',
        '--weights',
        help='trained weight files')

    argparser.add_argument(
        '-f',
        '--folder',
        help='folder with image files to run inference on')   

    args = argparser.parse_args()
    
    if args.create_dataset:
        from pascal_voc_writer import Writer
        
    with open(args.config) as config_buffer:
        config = json.loads(config_buffer.read())
    setup_inference(config, args.weights, args.threshold, args.folder)
