<h1 align="center">
  <img src="https://raw.githubusercontent.com/AIWintermuteAI/aXeleRate/master/resources/logo.png" alt="aXeleRate" width="350">
</h1>

<h3 align="center">Keras-based framework for AI on the Edge</h3>

<hr>
<p align="center">
aXeleRate streamlines training and converting computer vision models to be run on various platforms with hardware acceleration. It is optimized for both the workflow on local machine and on Google Colab. Currently supports trained model conversion to: .kmodel(K210), .tflite format(full integer and dymaic range quantization support available), OpenVINO IR model format. Experimental support: Google Edge TPU.
</p>

<table>
  <tr>
    <td>Standford Dog Breed Classification Dataset NASNetMobile backend + Classifier <a href="https://colab.research.google.com/github/AIWintermuteAI/aXeleRate/blob/master/resources/aXeleRate_standford_dog_classifier.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> </td>
     <td>PASCAL-VOC 2012 Object Detection Dataset MobileNet1_0 backend + YOLOv2 <a href="https://colab.research.google.com/github/AIWintermuteAI/aXeleRate/blob/master/resources/aXeleRate_pascal20_detector.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> </td>
     <td>Human parsing Semantic Segmentation MobileNet5_0 backend + Segnet-Basic <a href="https://colab.research.google.com/github/AIWintermuteAI/aXeleRate/blob/master/resources/aXeleRate_human_segmentation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> </td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/AIWintermuteAI/aXeleRate/master/resources/n02106550_7003.jpg" width=300 height=300></td>
    <td><img src="https://raw.githubusercontent.com/AIWintermuteAI/aXeleRate/master/resources/2009_001349.jpg" width=300 height=300></td>
    <td><img src="https://raw.githubusercontent.com/AIWintermuteAI/aXeleRate/master/resources/66.jpg" width=250 height=350></td>
  </tr>
 </table>

### aXeleRate

TL;DR

aXeleRate is meant for people who need to run computer vision applications(image classification, object detection, semantic segmentation) on the edge devices with hardware acceleration. It has easy configuration process through config file or config dictionary(for Google Colab) and automatic conversion of the best model for training session into the required file format. You put the properly formatted data in, start the training script and (hopefully) come back to see a converted model that is ready for deployment on your device!

### :wrench: Key Features
  - Supports multiple computer vision models: object detection(YOLOv2), image classification, semantic segmentation(SegNet-basic)
  - Different feature extractors to be used with the above network types: Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, NASNetMobile, ResNet50, and DenseNet121. 
  - Automatic conversion of the best model for the training session. aXeleRate will download the suitable converter automatically.
  - Currently supports trained model conversion to: .kmodel(K210), .tflite format(full integer and dymaic range quantization support available), .tflite(Edge TPU), .onnx(for later on-device optimization with TensorRT), OpenVINO IR model and .blob (for inference with Movidius chipes, such as OpenCV AI Kit).
  - Model version control made easier. Keras model files and converted models are saved in the project folder, grouped by the training date. Training history is saved as .png graph in the model folder.
  - Two modes of operation: locally, with train.py script and .json config file and remote, tailored for Google Colab, with module import and dictionary config.

### 💾 Install

Stable version:

pip install axelerate

Daily development version:

pip install git+https://github.com/AIWintermuteAI/aXeleRate

If installing in Anaconda environment, make sure you have necessary CUDA/CUDNN version installed in that environment to use GPU for training.

###  :computer: Project Story

aXeleRate started as a personal project of mine for training YOLOv2 based object detection networks and exporting them to .kmodel format to be run on K210 chip. I also needed to train image classification networks. And sometimes I needed to run inference with Tensorflow Lite on Raspberry Pi. As a result I had a whole bunch of disconnected scripts each had somewhat overlapping functionality. So, I decided to fix that and share the results with other people who might have similiar workflows.

aXeleRate is still work in progress project. I will be making some changes from time to time and if you find it useful and can contribute, PRs are very much welcome!

:ballot_box_with_check: TODO list:

TODO list is moving to Github Projects!

### Acknowledgements

  - YOLOv2 Keras code jeongjoonsup and Ngoc Anh Huynh https://github.com/experiencor/keras-yolo2 https://github.com/penny4860/Yolo-digit-detector
  - SegNet Keras code Divam Gupta https://github.com/divamgupta/image-segmentation-keras
  - Big Thank You to creator/maintainers of Keras/Tensorflow

### Donation
Recently there were a few people that wanted to make a small donation to aXeleRate, because it helped them with their work. I was caught off guard with the question about donations :) I didn't have anything set up, so I quickly created a page for them to be able to send money. If aXeleRate was useful in your work, you can donate a pizza or a beer to the project here https://www.buymeacoffee.com/hardwareai . But times are tough now(and always), so if you don't have much to spare, don't feel guilty! aXeleRate is totally open source and free to use.
