import argparse
import json
from axelerate import setup_inference
#from axelerate.networks.yolo.backend.utils.augment import visualize_dataset
from keras import backend as K 

argparser = argparse.ArgumentParser(description='Test axelerate on sample datasets')

argparser.add_argument(
    '-c',
    '--conf',
    default=None,
    help='path to configuration file')

argparser.add_argument(
    '-t',
    '--type',
    default="classifier",
    help='type of network to test:classifier,detector,segnet or all')

argparser.add_argument(
    '-w',
    '--weights',
    help='trained weight files')

args = argparser.parse_args()

def configs(network_type):

    classifier = {
        "model" : {
            "type":                 "Classifier",
            "architecture":         "MobileNet7_5",
            "input_size":           224,
            "fully-connected":      [100,50],
            "labels":               [],
            "dropout" : 		0.5
        },
        "pretrained" : {
            "full":   				""
        },
        "train" : {
            "actual_epoch":         5,
            "train_image_folder":   "sample_datasets/classifier/imgs",
            "train_times":          4,
            "valid_image_folder":   "sample_datasets/classifier/imgs_validation",
            "valid_times":          4,
            "batch_size":           4,
            "learning_rate":        1e-4,
            "saved_folder":   		"/home/ubuntu/space safety/classifier",
            "first_trainable_layer": "65",
            "augumentation":				True
        },
        "converter" : {
            "type":   				["k210","tflite"]
        }
    }


    detector = {
        "model":{
            "type":                 "Detector",
            "architecture":         "MobileNet7_5",
            "input_size":           224,
            "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
            "labels":               ["aeroplane","person","diningtable","bottle"],
            "coord_scale" : 		1.0,
            "class_scale" : 		1.0,
            "object_scale" : 		5.0,
            "no_object_scale" : 	1.0
        },
        "pretrained" : {
            "full":   				""
        },
        "train" : {
            "actual_epoch":         5,
            "train_image_folder":   "sample_datasets/detector/imgs",
            "train_annot_folder":   "sample_datasets/detector/anns",
            "train_times":          4,
            "valid_image_folder":   "sample_datasets/detector/imgs_validation",
            "valid_annot_folder":   "sample_datasets/detector/anns_validation",
            "valid_times":          4,
            "batch_size":           4,
            "learning_rate":        1e-4,
            "saved_folder":   		"/home/ubuntu/space safety/detector",
            "first_trainable_layer": "",
            "augumentation":				True,
            "is_only_detect" : 		False
        },
        "converter" : {
            "type":   				["k210","tflite"]
        }
    }

    segnet = {
            "model" : {
                "type":                 "SegNet",
                "architecture":         "MobileNet7_5",
                "input_size":           224,
                "n_classes" : 		21
            },
            "pretrained" : {
                "full":   				""
            },
            "train" : {
                "actual_epoch":         5,
                "train_image_folder":   "sample_datasets/segmentation/imgs",
                "train_annot_folder":   "sample_datasets/segmentation/anns",
                "train_times":          4,
                "valid_image_folder":   "sample_datasets/segmentation/imgs_validation",
                "valid_annot_folder":   "sample_datasets/segmentation/anns_validation",
                "valid_times":          4,
                "batch_size":           8,
                "learning_rate":        1e-4,
                "saved_folder":   		"/home/ubuntu/space safety/segment",
                "first_trainable_layer": "0",
                "ignore_zero_class":    False,
                "augumentation":				True
            },
            "converter" : {
                "type":   				["k210","tflite"]
            }
        }

    dict = {'classifier':[classifier],'detector':[detector],'segnet':[segnet]}

    return dict[network_type]

#visualize_dataset('/home/ubuntu/github/sample_datasets/detector/imgs','/home/ubuntu/github/sample_datasets/detector/anns')

if not args.conf:
    for item in configs(args.type):
        setup_inference(item,args.weights)
        K.clear_session()
else:
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())
        setup_inference(config,args.weights)

