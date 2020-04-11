# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import argparse
import os
import sys
import json
from axelerate.networks.yolo.frontend import create_yolo, get_object_labels
from axelerate.networks.classifier.frontend_classifier import create_classifier, get_labels
from axelerate.networks.segnet.frontend_segnet import create_segnet
from axelerate.networks.common_utils.convert import Converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default="configs/from_scratch.json",
    help='path to configuration file')

def train_from_config(config,project_folder):
    # Create the converter
    converter = Converter(config['converter']['type'])

    #  Segmentation network
    if config['model']['type']=='SegNet':
        print('Segmentation')           
        # 1. Construct the model 
        segnet = create_segnet(config['model']['architecture'],
                                   config['model']['input_size'],
                                   config['model']['n_classes'])   
        # 2. Load the pretrained weights (if any) 
        segnet.load_weights(config['pretrained']['full'], by_name=True)
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
                                           config['train']['ignore_zero_class'])
               
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
                                       config['model']['input_size'],
                                       config['model']['fully-connected'],
                                       config['model']['dropout'])   
        # 2. Load the pretrained weights (if any) 
        classifier.load_weights(config['pretrained']['full'], by_name=True)

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
                                               config['train']['first_trainable_layer'])



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
                           config['model']['input_size'],
                           config['model']['anchors'],
                           config['model']['coord_scale'],
                           config['model']['class_scale'],
                           config['model']['object_scale'],
                           config['model']['no_object_scale'])
        
        # 2. Load the pretrained weights (if any) 
        yolo.load_weights(config['pretrained']['full'], by_name=True)

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
                                           config['train']['first_trainable_layer'])
    converter.convert_model(model_path,model_layers,config['train']['valid_image_folder'])    
    return model_path

def setup_training(config_file=None,config_dict=None):
    """make directory to save weights & its configuration """
    if config_file:
        with open(config_file) as config_buffer:
            config = json.loads(config_buffer.read())
    elif config_dict:
        config = config_dict
    else:
        print('No config found')
        sys.exit()
    dirname = config['train']['saved_folder']
    if os.path.isdir(dirname):
        print("Project folder {} already exists. Creating a folder for new training session.".format(dirname))
    else:
        print("Project folder {} is created.".format(dirname, dirname))
        os.makedirs(dirname)
    #print("Weight file and Config file will be saved in \"{}\"".format(dirname))
    return(train_from_config(config, dirname))


if __name__ == '__main__':
    args = argparser.parse_args()
    setup_training(args.conf)
