# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337)
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2

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

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes_ = np.copy(boxes)
        labels_ = np.copy(labels)
  
        # 2. resize and augment image     
        image, boxes_, labels_ = process_image(image, boxes_, labels_, self._w, self._h, self._jitter) 

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


def process_image(image, boxes, labels, desired_w, desired_h, augment):
    
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
            #iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            #iaa.Flipud(0.2),  # vertically flip 20% of all images
            # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent
                rotate=(-15, 15),  # rotate by -45 to +45 degrees
                shear=(-15, 15),  # shear by -16 to +16 degrees
                # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 3),
                       [
                           # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means (kernel sizes between 2 and 7)
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians (kernel sizes between 2 and 7)
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                           # search either for all edges or for directed edges
                           # sometimes(iaa.OneOf([
                           #    iaa.EdgeDetect(alpha=(0, 0.7)),
                           #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                           # ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           #iaa.Invert(0.05, per_channel=True), # invert color channels
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


def visualize_dataset(img_folder, ann_folder, num_imgs = None, img_size=None, jitter=None):
    import os
    from axelerate.networks.yolo.backend.utils.annotation import PascalVocXmlParser
    import matplotlib.pyplot as plt
    parser = PascalVocXmlParser()
    for ann in os.listdir(ann_folder)[:num_imgs]:
        annotation_file = os.path.join(ann_folder, ann)
        fname = parser.get_fname(annotation_file)
        labels = parser.get_labels(annotation_file)
        boxes = parser.get_boxes(annotation_file)
        img_file =  os.path.join(img_folder, fname)

        aug = ImgAugment(img_size, img_size, jitter=jitter)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--num_imgs", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--aug", type=bool)
    args = parser.parse_args()
    visualize_dataset(args.images, args.annotations, args.num_imgs, args.img_size, args.aug)

