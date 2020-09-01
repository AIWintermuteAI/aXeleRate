# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import argparse
import os
import sys
import json
import matplotlib

from axelerate.networks.yolo.frontend import create_yolo, get_object_labels
from axelerate.networks.classifier.frontend_classifier import create_classifier, get_labels
from axelerate.networks.segnet.frontend_segnet import create_segnet
from axelerate.networks.common_utils.convert import Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/from_scratch.json",
    help='path to configuration file')

def train_from_config(config,project_folder):
    try:
        matplotlib.use('Agg')
    except:
        pass

    #added for compatibility with < 0.5.7 versions
    try:
        input_size = config['model']['input_size'][:]
    except:
        input_size = [config['model']['input_size'],config['model']['input_size']]

    # Create the converter
    converter = Converter(config['converter']['type'], config['model']['architecture'], config['train']['valid_image_folder'])

    #  Segmentation network
    if config['model']['type']=='SegNet':
        print('Segmentation')           
        # 1. Construct the model 
        segnet = create_segnet(config['model']['architecture'],
                                   input_size,
                                   config['model']['n_classes'],
                                   config['weights']['backend'])   
        # 2. Load the pretrained weights (if any) 
        segnet.load_weights(config['weights']['full'], by_name=True)
        # 3. actual training 
        model_layers, model_path = segnet.train(config['train']['train_image_folder'],
                                           config['train']['train_annot_folder'],
                                           config['train']['actual_epoch'],
                                           project_folder,
                                           config["train"]["batch_size"],
                                           config["train"]["augumentation"],
                                           config['train']['learning_rate'], 
                                           config['train']['train_times'],
                                           config['train']['valid_times'],
                                           config['train']['valid_image_folder'],
                                           config['train']['valid_annot_folder'],
                                           config['train']['first_trainable_layer'],
                                           config['train']['ignore_zero_class'],
                                           config['train']['valid_metric'])
               
    #  Classifier
    if config['model']['type']=='Classifier':
        print('Classifier')           
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_labels(config['train']['train_image_folder'])
                 # 1. Construct the model 
        classifier = create_classifier(config['model']['architecture'],
                                       labels,
                                       input_size,
                                       config['model']['fully-connected'],
                                       config['model']['dropout'],
                                       config['weights']['backend'],
                                       config['weights']['save_bottleneck'])   
        # 2. Load the pretrained weights (if any) 
        classifier.load_weights(config['weights']['full'], by_name=True)

        # 3. actual training 
        model_layers, model_path = classifier.train(config['train']['train_image_folder'],
                                               config['train']['actual_epoch'],
                                               project_folder,
                                               config["train"]["batch_size"],
                                               config["train"]["augumentation"],
                                               config['train']['learning_rate'], 
                                               config['train']['train_times'],
                                               config['train']['valid_times'],
                                               config['train']['valid_image_folder'],
                                               config['train']['first_trainable_layer'],
                                               config['train']['valid_metric'])



    #  Detector
    if config['model']['type']=='Detector':
        if config['train']['is_only_detect']:
            labels = ["object"]
        else:
            if config['model']['labels']:
                labels = config['model']['labels']
            else:
                labels = get_object_labels(config['train']['train_annot_folder'])
        print(labels)

        # 1. Construct the model 
        yolo = create_yolo(config['model']['architecture'],
                           labels,
                           input_size,
                           config['model']['anchors'],
                           config['model']['coord_scale'],
                           config['model']['class_scale'],
                           config['model']['object_scale'],
                           config['model']['no_object_scale'],
                           config['weights']['backend'])
        
        # 2. Load the pretrained weights (if any) 
        yolo.load_weights(config['weights']['full'], by_name=True)

        # 3. actual training 
        model_layers, model_path = yolo.train(config['train']['train_image_folder'],
                                           config['train']['train_annot_folder'],
                                           config['train']['actual_epoch'],
                                           project_folder,
                                           config["train"]["batch_size"],
                                           config["train"]["augumentation"],
                                           config['train']['learning_rate'], 
                                           config['train']['train_times'],
                                           config['train']['valid_times'],
                                           config['train']['valid_image_folder'],
                                           config['train']['valid_annot_folder'],
                                           config['train']['first_trainable_layer'],
                                           config['train']['valid_metric'])
    # 4 Convert the model
    converter.convert_model(model_path)    
    return model_path

def setup_training(config_file=None, config_dict=None):
    """make directory to save weights & its configuration """
    if config_file:
        with open(config_file) as config_buffer:
            config = json.loads(config_buffer.read())
    elif config_dict:
        config = config_dict
    else:
        print('No config found')
        sys.exit()
    dirname = os.path.join("projects", config['train']['saved_folder'])
    if os.path.isdir(dirname):
        print("Project folder {} already exists. Creating a folder for new training session.".format(dirname))
    else:
        print("Project folder {} is created.".format(dirname, dirname))
        os.makedirs(dirname)
    #print("Weight file and Config file will be saved in \"{}\"".format(dirname))
    return(train_from_config(config, dirname))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='Train and validate YOLO_v2 model on any dataset')

    argparser.add_argument(
        '-c',
        '--config',
        default="configs/classifer.json",
        help='path to configuration file')

    args = argparser.parse_args()
    setup_training(config_file=args.config)
