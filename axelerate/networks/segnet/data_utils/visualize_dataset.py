#!/usr/bin/env python

import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

from axelerate.networks.segnet.data_utils.augmentation import augment_seg
from axelerate.networks.segnet.data_utils.data_loader import get_pairs_from_paths, DATA_LOADER_SEED, class_colors, DataLoaderError

random.seed(DATA_LOADER_SEED)

def _get_colored_segmentation_image(img, seg, colors, n_classes, do_augment=False):
    """ Return a colored segmented image """
    seg_img = np.zeros_like(seg)

    if do_augment:
        img, seg[:, :, 0] = augment_seg(img, seg[:, :, 0])

    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                            (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                            (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                            (colors[c][2])).astype('uint8')

    return img , seg_img


def visualize_segmentation_dataset(images_path, segs_path, n_classes,
                                   do_augment=False, ignore_non_matching=False,
                                   no_show=False):
    try:
        # Get image-segmentation pairs
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path,
                            ignore_non_matching=ignore_non_matching)

        # Get the colors for the classes
        colors = class_colors

        print("Please press any key to display the next image")
        for im_fn, seg_fn in img_seg_pairs:
            img = cv2.imread(im_fn)[...,::-1]
            seg = cv2.imread(seg_fn)
            print("Found the following classes in the segmentation image:", np.unique(seg))
            img , seg_img = _get_colored_segmentation_image(img, seg, colors, n_classes, do_augment=do_augment)
            #print("Please press any key to display the next image")
            #cv2.imshow("img", img)
            #cv2.imshow("seg_img", seg_img)
            #cv2.waitKey()
            fig = plt.figure(figsize=(14,7))
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(img)
            ax2 = fig.add_subplot(1,3,2)
            ax2.imshow(seg)
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(seg_img)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def visualize_segmentation_dataset_one(images_path, segs_path, n_classes,
                                       do_augment=False, no_show=False, ignore_non_matching=False):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path, ignore_non_matching=ignore_non_matching)

    colors = class_colors

    im_fn, seg_fn = random.choice(img_seg_pairs)

    img = cv2.imread(im_fn)
    seg = cv2.imread(seg_fn)
    print("Found the following classes in the segmentation image:", np.unique(seg))

    img,seg_img = _get_colored_segmentation_image(img, seg, colors,n_classes, do_augment=do_augment)

    if not no_show:
        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cv2.waitKey()

    return img, seg_img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str)
    parser.add_argument("--annotations", type=str)
    parser.add_argument("--n_classes", type=int)
    args = parser.parse_args()

    visualize_segmentation_dataset(args.images, args.annotations, args.n_classes)
