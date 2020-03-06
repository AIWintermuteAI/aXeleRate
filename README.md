# Forked from SVHN yolo-v2 digit detector

![Raccoon](https://cdn.instructables.com/FUW/ZFON/K0WP7IHC/FUWZFONK0WP7IHC.LARGE.jpg?auto=webp&width=1024&fit=bounds)

## Usage for python code

#### 0. Requirement

* python 3.6
* tensorflow 1.14.0
* keras 2.2.4
* opencv 3.3.0
* Etc.

I recommend that you create and use an anaconda env that is independent of your project. You can create anaconda env for this project by following these simple steps. It is recomended that you use Ubuntu (16.04 or 18.04) for this project - you can train the model on Windows, but for conversion step you will need a Linux computer.

```
$ conda create -n yolo python=3.6
$ activate yolo # in linux "source activate yolo"
(yolo) $ pip install -r requirements.txt
(yolo) $ pip install -e .
```

### 1. Training from scratch

This project provides a way to train digit detector from scratch. If you follow the command below, you can build a digit detector with just two images.


* First, train all layers through the following command(change from_scratch.json to the name of your config file. Use MobileNet as feature extractor). 
  * `` project/root> python train.py -c configs/from_scratch.json ``
* Then, evaluate trained digit detector(change the -w argument with the location of your weights)
  * `` project/root> python evaluate.py -c configs/from_scratch.json -w svhn/weights.h5 ``
  * The prediction result images are saved in the ``project/detected`` directory.

Now you can add more images to train a detector with good generalization performance.

### 3. SVHN dataset in Pascal Voc annotation format

In this project, [pascal voc format](http://host.robots.ox.ac.uk/pascal/VOC/) is used as annotation information to train object detector.
An annotation file of this format can be downloaded from [svhn-voc-annotation-format](https://github.com/penny4860/svhn-voc-annotation-format).


### 1. Raccoon dataset : https://github.com/experiencor/raccoon_dataset

![Raccoon](https://cdn.instructables.com/FT9/9YL1/K0WPA4SD/FT99YL1K0WPA4SD.LARGE.jpg?auto=webp&width=1024&height=1024&fit=bounds)

## Copyright

* See [LICENSE](LICENSE) for details.
* This project started at [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras).  penny4860 refactored the source code structure of [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras) and added the CI test.  penny4860 also applied the SVHN dataset to implement the digit detector. Thanks to the [Huynh Ngoc Anh](https://github.com/experiencor) for providing a good project as open source.

## See Also

If you are interested in other projects with Kendryte K210 chip, please refer to the following projects. 

* https://github.com/AIWintermuteAI/transfer_learning_sipeed
	* Using MobileNet for image recognition

* https://github.com/AIWintermuteAI/maixpy-openmv-demos
	* OpenMV demos for micropython firmware, including detecting shapes, faces, colors, etc.

* https://github.com/AIWintermuteAI/kendryte-standalone-demo
	* A demo for Kendryte Standalone SDK, detecting objects with YOLO and passing the class of the object to another device using UART

