import os
import numpy as np
np.random.seed(1337)
from keras.utils import Sequence
from axelerate.networks.common_utils.augment import process_image_segmentation
import glob
import itertools
import random
import six
import cv2

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")
    def tqdm(iter):
        return iter


from ..models.config import IMAGE_ORDERING

DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


class DataLoaderError(Exception):
    pass

def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png" , ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(file_name, os.path.join(segs_path, dir_entry)))
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation found for image {0}.".format(image_full_path))

    return return_value


def get_image_array(image_input, norm, ordering='channels_first'):
    """ Load image array from input """
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif  isinstance(image_input, six.string_types)  :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))
        
    if norm:
        img = norm(img)

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nClasses, no_reshape=True):
    """ Load segmentation array from input """

    seg_labels = np.zeros((image_input.shape[0], image_input.shape[1], nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types) :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: Can't process input type {0}".format(str(type(image_input))))

    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path, n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: {0} and segmentations path: {1}".format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} doesn't match (possibly the files are corrupt).".format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} violating range [0, {1}]. Found maximum pixel value {2}".format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False
        
        
def create_batch_generator(images_path, segs_path, 
                           input_size=224,
                           output_size=112,
                           n_classes=51,
                           batch_size=8,
                           repeat_times=1,
                           do_augment=False,
                           norm=None):

    worker = BatchGenerator(images_path, segs_path, batch_size,
                 n_classes, input_size, output_size, repeat_times, 
                 do_augment, norm)
    return worker


class BatchGenerator(Sequence):
    def __init__(self,
                 images_path, segs_path, batch_size,
                 n_classes,input_size, output_size, repeat_times,
                 do_augment=False, norm=None):
        self.norm = norm
        self.n_classes = n_classes
        self.input_size = input_size
        self.output_size = output_size
        self.do_augment = do_augment
        self._repeat_times = repeat_times
        self._batch_size = batch_size
        self.img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        random.shuffle(self.img_seg_pairs)
        self.zipped = itertools.cycle(self.img_seg_pairs)
        self.counter = 0

    def __len__(self):
        return int(len(self.img_seg_pairs) * self._repeat_times/self._batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        x_batch = []
        y_batch= []
        for i in range(self._batch_size):
            img, seg = next(self.zipped)
            img = cv2.imread(img, 1)[...,::-1]
            seg = cv2.imread(seg, 1)

            im, seg = process_image_segmentation(img, seg, self.input_size[0], self.input_size[1], self.output_size[0], self.output_size[1], self.do_augment)

            x_batch.append(get_image_array(im, self.norm, ordering=IMAGE_ORDERING))
            y_batch.append(get_segmentation_array(seg, self.n_classes))

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        self.counter += 1
        return x_batch, y_batch

    def on_epoch_end(self):
        self.counter = 0
        random.shuffle(self.img_seg_pairs)
