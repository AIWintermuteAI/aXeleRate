# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337)
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib

class ImgAugment(object):
    def __init__(self, w, h, jitter):
        """
        # Args
            desired_w : int
            desired_h : int
            jitter : bool
        """
        self._jitter = jitter
        self._w = w
        self._h = h

    def imread(self, img_file, boxes, labels):
        """
        # Args
            img_file : str
            boxes : array, shape of (N, 4)
        
        # Returns
            image : 3d-array, shape of (h, w, 3)
            boxes_ : array, same shape of boxes
                jittered & resized bounding box
        """
        # 1. read image file
        try:
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print("This image has an annotation file, but cannot be open. Check the integrity of your dataset.", img_file)
            raise
        

        boxes_ = np.copy(boxes)
        labels_ = np.copy(labels)
  
        # 2. resize and augment image     
        image, boxes_, labels_ = process_image_detection(image, boxes_, labels_, self._w, self._h, self._jitter) 

        return image, boxes_, labels_


def _to_bbs(boxes, labels, shape):
    new_boxes = []
    for i in range(len(boxes)):
        x1,y1,x2,y2 = boxes[i]
        new_box = BoundingBox(x1,y1,x2,y2, labels[i])
        new_boxes.append(new_box)
    bbs = BoundingBoxesOnImage(new_boxes, shape)
    return bbs

def _to_array(bbs):
    new_boxes = []
    new_labels = []
    for bb in bbs.bounding_boxes:
        x1 = int(bb.x1)
        x2 = int(bb.x2)
        y1 = int(bb.y1)
        y2 = int(bb.y2)
        label = bb.label
        new_boxes.append([x1,y1,x2,y2])
        new_labels.append(label)
    return new_boxes, new_labels


def process_image_detection(image, boxes, labels, desired_w, desired_h, augment):
    
    # resize the image to standard size
    if (desired_w and desired_h) or augment:
        bbs = _to_bbs(boxes, labels, image.shape)

        if (desired_w and desired_h):
            # Rescale image and bounding boxes
            image = ia.imresize_single_image(image, (desired_w, desired_h))
            bbs = bbs.on(image)

        if augment:
            aug_pipe = _create_augment_pipeline()
            image, bbs = aug_pipe(image=image, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()

        new_boxes, new_labels = _to_array(bbs)
        #if len(new_boxes) != len(boxes):
        #    print(new_boxes)
        #    print(boxes)
        #    print("_________________")

        return image, np.array(new_boxes), new_labels
    else:
        return image, np.array(boxes), labels

def process_image_classification(image, desired_w, desired_h, augment):
    
    # resize the image to standard size
    if (desired_w and desired_h) or augment:

        if (desired_w and desired_h):
            # Rescale image
            image = ia.imresize_single_image(image, (desired_w, desired_h))

        if augment:
            aug_pipe = _create_augment_pipeline()
            image = aug_pipe(image=image)
        
    return image

def process_image_segmentation(image, segmap, input_w, input_h, output_w, output_h, augment):
    # resize the image to standard size
    if (input_w and input_h) or augment:
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

        if (input_w and input_h):
            # Rescale image and segmaps
            image = ia.imresize_single_image(image, (input_w, input_h))
            segmap = segmap.resize((output_w, output_h), interpolation="nearest")

        if augment:
            aug_pipe = _create_augment_pipeline()
            image, segmap = aug_pipe(image=image, segmentation_maps=segmap)

    return image, segmap.get_arr()


def _create_augment_pipeline():
    
    # augmentors by https://github.com/aleju/imgaug
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    aug_pipe = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            #iaa.Flipud(0.2),  # vertically flip 20% of all images

            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                       [iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},),
                        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},),
                        iaa.Affine(rotate=(-15, 15),),
                        iaa.Affine(shear=(-15, 15)),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means (kernel sizes between 2 and 7)
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians (kernel sizes between 2 and 7)
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    return aug_pipe


def visualize_detection_dataset(img_folder, ann_folder, num_imgs = None, img_size=None, jitter=None):
    from axelerate.networks.yolo.backend.utils.annotation import PascalVocXmlParser
    try:
        matplotlib.use('TkAgg')
    except:
        pass

    parser = PascalVocXmlParser()
    aug = ImgAugment(img_size, img_size, jitter=jitter)
    for ann in os.listdir(ann_folder)[:num_imgs]:
        annotation_file = os.path.join(ann_folder, ann)
        fname = parser.get_fname(annotation_file)
        labels = parser.get_labels(annotation_file)
        boxes = parser.get_boxes(annotation_file)
        img_file =  os.path.join(img_folder, fname)
        img, boxes_, labels_ = aug.imread(img_file, boxes, labels)
        
        for i in range(len(boxes_)):
            x1, y1, x2, y2 = boxes_[i]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(img, 
                        '{}'.format(labels_[i]), 
                        (x1, y1 - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * img.shape[0], 
                        (255,0,0), 1)

        plt.imshow(img)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def visualize_segmentation_dataset(images_path, segs_path, num_imgs = None, img_size=None, do_augment=False, n_classes=255):

    from axelerate.networks.segnet.data_utils.data_loader import get_pairs_from_paths, DATA_LOADER_SEED, class_colors, DataLoaderError

    try:
        matplotlib.use('TkAgg')
    except:
        pass

    def _get_colored_segmentation_image(img, seg, colors, n_classes, img_size, do_augment=False):
        """ Return a colored segmented image """

        img, seg = process_image_segmentation(img, seg, img_size, img_size, img_size, img_size, do_augment)
        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                                (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                                (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                                (colors[c][2])).astype('uint8')

        return img, seg_img

    try:
        # Get image-segmentation pairs
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, ignore_non_matching=True)
        # Get the colors for the classes
        colors = class_colors

        print("Please press any key to display the next image")
        for im_fn, seg_fn in img_seg_pairs[:num_imgs]:
            img = cv2.imread(im_fn)[...,::-1]
            seg = cv2.imread(seg_fn)
            print("Found the following classes in the segmentation image:", np.unique(seg))
            img, seg_img = _get_colored_segmentation_image(img, seg, colors, n_classes, img_size, do_augment=do_augment)
            fig = plt.figure(figsize=(14,7))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(img)
            ax3 = fig.add_subplot(1,2,2)
            ax3.imshow(seg_img)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False

def visualize_classification_dataset(img_folder, num_imgs = None, img_size=None, jitter=None):
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_files_list = []
    image_search = lambda ext : glob.glob(img_folder + ext, recursive=True)
    for ext in ['/**/*.jpg', '/**/*.jpeg', '/**/*.png']: image_files_list.extend(image_search(ext))
    random.shuffle(image_files_list)
    for filename in image_files_list[0:num_imgs]:
        image = cv2.imread(filename)[...,::-1]
        image = process_image_classification(image, img_size, img_size, jitter)
        cv2.putText(image, os.path.dirname(filename).split('/')[-1], (10,30), font, image.shape[1]/700 , (255, 0, 0), 2, True)
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print(filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--num_imgs", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--aug", type=bool)
    args = parser.parse_args()
    if args.type == 'detection':
        visualize_detection_dataset(args.images, args.annotations, args.num_imgs, args.img_size, args.aug)
    if args.type == 'segmentation':
        visualize_segmentation_dataset(args.images, args.annotations, args.num_imgs, args.img_size, args.aug)
    if args.type == 'classification':
        visualize_classification_dataset(args.images, args.num_imgs, args.img_size, args.aug)
