import argparse
from axelerate import setup_training, setup_inference
from keras import backend as K 

argparser = argparse.ArgumentParser(description='Test axelerate on sample datasets')

argparser.add_argument(
    '-t',
    '--type',
    default="all",
    help='type of network to test:classifier,detector,segnet or all')

args = argparser.parse_args()

def configs(network_type):

    classifier = {
        "model" : {
            "type":                 "Classifier",
            "architecture":         "VGG16",
            "input_size":           224,
            "fully-connected":      [100,50],
            "labels":               [],
            "dropout" : 		0.5
        },
        "pretrained" : {
            "full":   				""
        },
        "train" : {
            "actual_epoch":         1,
            "train_image_folder":   "sample_datasets/classifier/imgs",
            "train_times":          4,
            "valid_image_folder":   "sample_datasets/classifier/imgs_validation",
            "valid_times":          4,
            "batch_size":           4,
            "learning_rate":        1e-4,
            "saved_folder":   		"/home/ubuntu/space safety/classifier",
            "first_trainable_layer": "",
            "augumentation":				True
        },
        "converter" : {
            "type":   				["tflite"]
        }
    }


    detector = {
        "model":{
            "type":                 "Detector",
            "architecture":         "SqueezeNet",
            "input_size":           224,
            "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
            "labels":               ["aeroplane","person","diningtable","bottle","bird","bus","boat","cow","sheep","train"],
            "coord_scale" : 		1.0,
            "class_scale" : 		1.0,
            "object_scale" : 		5.0,
            "no_object_scale" : 	1.0
        },
        "pretrained" : {
            "full":   				""
        },
        "train" : {
            "actual_epoch":         1,
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
            "type":   				["tflite"]
        }
    }

    segnet = {
            "model" : {
                "type":                 "SegNet",
                "architecture":         "MobileNet2_5",
                "input_size":           224,
                "n_classes" : 		20
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
                "first_trainable_layer": "",
                "ignore_zero_class":    False,
                "augumentation":				True
            },
            "converter" : {
                "type":   				["k210","tflite"]
            }
        }

    dict = {'all':[classifier,detector,segnet],'classifier':[classifier],'detector':[detector],'segnet':[segnet]}

    return dict[network_type]

for item in configs(args.type):
    model_path = setup_training(config_dict=item)
    K.clear_session()
    setup_inference(item,model_path)



