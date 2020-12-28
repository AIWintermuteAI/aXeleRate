import argparse
import json
from axelerate import setup_training, setup_evaluation
import tensorflow.keras.backend as K
from termcolor import colored
import traceback
import time 

def configs(network_type):

    classifier = {
        "model" : {
            "type":                 "Classifier",
            "architecture":         "Tiny Yolo",
            "input_size":           [224,224],
            "fully-connected":      [],
            "labels":               [],
            "dropout" : 		    0.5
        },
        "weights" : {
            "full":   				"",
            "backend":   		    None,
            "save_bottleneck":      True
        
        },
        "train" : {
            "actual_epoch":         5,
            "train_image_folder":   "sample_datasets/classifier/imgs",
            "train_times":          1,
            "valid_image_folder":   "sample_datasets/classifier/imgs_validation",
            "valid_times":          1,
            "valid_metric":         "val_accuracy",
            "batch_size":           2,
            "learning_rate":        1e-4,
            "saved_folder":   		"classifier",
            "first_trainable_layer": "",
            "augumentation":		True
        },
        "converter" : {
            "type":   				[]
        }
    }


    detector = {
        "model":{
            "type":                 "Detector",
            "architecture":         "MobileNet7_5",
            "input_size":           224,
            "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
            "labels":               ["aeroplane","person","diningtable","bottle","bird","bus","boat","cow","sheep","train"],
            "coord_scale" : 		1.0,
            "class_scale" : 		1.0,
            "object_scale" : 		5.0,
            "no_object_scale" : 	1.0
        },
        "weights" : {
            "full":   				"",
            "backend":   		    None
        },
        "train" : {
            "actual_epoch":         5,
            "train_image_folder":   "sample_datasets/detector/imgs",
            "train_annot_folder":   "sample_datasets/detector/anns",
            "train_times":          1,
            "valid_image_folder":   "sample_datasets/detector/imgs_validation",
            "valid_annot_folder":   "sample_datasets/detector/anns_validation",
            "valid_times":          1,
            "valid_metric":         "mAP",
            "batch_size":           2,
            "learning_rate":        1e-4,
            "saved_folder":   		"detector",
            "first_trainable_layer": "",
            "augumentation":		True,
            "is_only_detect" : 		False
        },
        "converter" : {
            "type":   				[]
        }
    }

    segnet = {
            "model" : {
                "type":                 "SegNet",
                "architecture":         "MobileNet5_0",
                "input_size":           [224,224],
                "n_classes" : 		20
            },
        "weights" : {
            "full":   				"",
            "backend":   		    None
        },
            "train" : {
                "actual_epoch":         5,
                "train_image_folder":   "sample_datasets/segmentation/imgs",
                "train_annot_folder":   "sample_datasets/segmentation/anns",
                "train_times":          4,
                "valid_image_folder":   "sample_datasets/segmentation/imgs_validation",
                "valid_annot_folder":   "sample_datasets/segmentation/anns_validation",
                "valid_times":          4,
                "valid_metric":         "val_loss",
                "batch_size":           2,
                "learning_rate":        1e-4,
                "saved_folder":   		"segment",
                "first_trainable_layer": "",
                "ignore_zero_class":    False,
                "augumentation":		True
            },
            "converter" : {
                "type":   				[]
            }
        }

    dict = {'all':[classifier,detector,segnet],'classifier':[classifier],'detector':[detector],'segnet':[segnet]}

    return dict[network_type]


argparser = argparse.ArgumentParser(description='Test axelerate on sample datasets')

argparser.add_argument(
    '-t',
    '--type',
    default="all",
    help='type of network to test:classifier,detector,segnet or all')
    
argparser.add_argument(
    '-a',
    '--arch',
    type=bool,
    default=False,
    help='test all architectures?')

argparser.add_argument(
    '-c',
    '--conv',
    type=bool,
    default=False,
    help='test all converters?')

args = argparser.parse_args()

archs = ['MobileNet7_5']
converters = [""]
errors = []

if args.arch:
    archs = ['Full Yolo', 'Tiny Yolo', 'MobileNet1_0', 'MobileNet7_5', 'MobileNet5_0', 'MobileNet2_5', 'SqueezeNet', 'NASNetMobile', 'ResNet50', 'DenseNet121']
if args.conv:
    converters = ['k210', 'tflite_fullint', 'tflite_dynamic', 'edgetpu', 'openvino', 'onnx']

for item in configs(args.type):
    for arch in archs:
        for converter in converters:
            try:
                item['model']['architecture'] = arch
                item['converter']['type'] = converter
                print(json.dumps(item, indent=4, sort_keys=False))
                model_path = setup_training(config_dict=item)
                K.clear_session()
                setup_evaluation(item, model_path)
            except Exception as e:
                traceback.print_exc()
                print(colored(str(e), 'red'))
                time.sleep(2)
                errors.append(item['model']['type'] + " " + arch + " " + converter + " " + str(e))

for error in errors:
    print(error)



