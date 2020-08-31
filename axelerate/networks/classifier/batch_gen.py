## Code heavily adapted from:
## *https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/

"""Utilities for real-time data augmentation on image data. """

from .directory_iterator import DirectoryIterator
from axelerate.networks.common_utils.augment import process_image_classification
from keras.utils import Sequence
import cv2
import os

def create_datagen(train_folder, valid_folder, batch_size, input_size, project_folder, augumentation, norm):

    train_datagen=ImageDataAugmentor(preprocess_input=  norm, process_image = process_image_classification, augment = augumentation)
    validation_datagen=ImageDataAugmentor(preprocess_input = norm, process_image = process_image_classification, augment = False)
    
    train_generator=train_datagen.flow_from_directory(train_folder,
                                                     target_size=input_size,
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='categorical', 
			                                         shuffle=True)

    validation_generator=validation_datagen.flow_from_directory(valid_folder,
                                                     target_size=input_size,
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='categorical', 
			                                         shuffle=True)		
				                                         
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    fo = open(os.path.join(project_folder,"labels.txt"), "w")
    for k,v in labels.items():
        print(v)
        fo.write(v+"\n")
    fo.close()
    return train_generator, validation_generator
    
    
class ImageDataAugmentor(Sequence):
    """Generate batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).
    # Arguments
        preprocessing_input: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image, and should output a Numpy tensor with the same shape.
        augment: augmentations passed as albumentations or imgaug transformation 
            or sequence of transformations.     
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 augment = False,
                 process_image=None,
                 preprocess_input=None,
                 data_format='channels_last'):
               
        self.augment = augment
        self.process_image = process_image
        self.preprocess_input = preprocess_input

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation=cv2.INTER_NEAREST):
        """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "gray", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataAugmentor`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )
    

    def transform_image(self, image, desired_w, desired_h):
        """
        Transforms an image by first augmenting and then standardizing
        """
        image = self.process_image(image, desired_w, desired_h, self.augment)
        image = self.preprocess_input(image)
        
        return image
