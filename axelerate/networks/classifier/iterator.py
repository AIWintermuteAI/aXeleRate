"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import numpy as np
from keras_preprocessing import get_keras_submodule
import matplotlib.pyplot as plt

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object

from .utils import (array_to_img,
                    img_to_array,
                    load_img)


class Iterator(IteratorType):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batch_of_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class BatchFromFilesMixin():
    """Adds methods related to getting batches from filenames

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             target_size,
                             color_mode,
                             data_format,
                             save_to_dir,
                             save_prefix,
                             save_format,
                             subset,
                             interpolation):
        """Sets attributes to use later for processing files into a batch.

        # Arguments
            image_data_generator: Instance of `ImageDataAugmentor`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"gray"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataAugmentor.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"cv2.INTER_NEAREST"`, `"cv2.INTER_LINEAR"`, `"cv2.INTER_AREA"`, `"cv2.INTER_CUBIC"`
                and `"cv2.INTER_LANCZOS4"`
                By default, `"cv2.INTER_NEAREST"` is used.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'gray'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "gray".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batch_of_samples(self, index_array, apply_standardization=True):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        
        # build batch of image data
        batch_x = np.array([load_img(filepaths[x], 
                                     color_mode=self.color_mode,
                                     target_size=self.target_size, 
                                     interpolation=self.interpolation) for x in index_array])    

        # apply the augmentations and custom transformations to the image data
        batch_x = np.array([self.image_data_generator.transform_image(x, self.target_size[0], self.target_size[1]) for x in batch_x])

        # transform to `channels_first` format if needed
        if self.data_format == "channels_first":
            batch_x = np.array([np.swapaxes(x,0,2) for x in batch_x])

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
            
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    def _get_batches_of_transformed_samples(self, index_array):
        return self._get_batch_of_samples(index_array)


    def show_batch(self, rows:int=5, apply_standardization:bool=False, **plt_kwargs):
        img_arr = np.random.choice(range(len(self.classes)), rows**2)
        if self.class_mode is None:
            imgs = self._get_batch_of_samples(img_arr, apply_standardization=apply_standardization)
        else:
            imgs, _ = self._get_batch_of_samples(img_arr, apply_standardization=apply_standardization)
            lbls = np.array(self.labels)[img_arr]
        
            try:
                inv_class_indices = {v: k for k, v in self.class_indices.items()}
                lbls = [inv_class_indices.get(k) for k in lbls]
            except:
                pass

        if self.data_format == "channels_first":
            imgs = np.array([np.swapaxes(img,0,2) for img in imgs])

        if not 'figsize' in plt_kwargs:
            plt_kwargs['figsize'] = (12,12)

        plt.close('all')
        plt.figure(**plt_kwargs)

        for idx, img in enumerate(imgs):
            plt.subplot(rows, rows, idx+1)
            plt.imshow(img.squeeze())
            if lbls is not None:
                plt.title(lbls[idx])
            plt.axis('off')
        
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()
        
    @property
    def filepaths(self):
        """List of absolute paths to image files"""
        raise NotImplementedError(
            '`filepaths` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation"""
        raise NotImplementedError(
            '`labels` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            '`sample_weight` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )
