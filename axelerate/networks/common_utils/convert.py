import tensorflow as tf
import subprocess
import os
import cv2
import keras.backend as K
import tarfile
import glob
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import urllib.request
from keras.utils import get_file
from keras.models import load_model

k210_converter_path=os.path.join(os.path.dirname(__file__),"ncc","ncc")
k210_converter_download_path=os.path.join(os.path.dirname(__file__),'ncc_linux_x86_64.tar.xz')
nncase_download_url="https://github.com/kendryte/nncase/releases/download/v0.2.0-beta2/ncc_linux_x86_64.tar.xz"
cwd = os.path.dirname(os.path.realpath(__file__))

class Converter(object):
    def __init__(self,converter_type,backend=None,img_size=None,dataset_path=None):
        if 'tflite' in converter_type:
            print('Tflite Converter ready')

        if 'k210' in converter_type:
            if os.path.exists(k210_converter_path):
                print('K210 Converter ready')
            else:
                print('Downloading K210 Converter')
                get_file(k210_converter_download_path,nncase_download_url)         
                tar_file = tarfile.open(k210_converter_download_path)
                tar_file.extractall(os.path.join(os.path.dirname(__file__),"ncc"))
                tar_file.close()
                os.chmod(k210_converter_path, 0o775)

        if 'edgetpu' in converter_type:
            rc, out = subprocess.getstatusoutput('dpkg -l edgetpu-compiler')
            if rc == 0:
                print('Edge TPU Converter ready')
            else:
                print('Installing Edge TPU Converter')
                subprocess.Popen(['bash install_edge_tpu_compiler.sh'], shell=True, stdin=subprocess.PIPE, cwd=cwd).communicate()
        self._converter_type = converter_type
        self._backend = backend
        self._dataset_path=dataset_path
        self._img_size = img_size

    def representative_dataset_gen(self):
        num_imgs = 10
        from axelerate.networks.common_utils.feature import create_feature_extractor
        backend = create_feature_extractor(self._backend,224)
        image_files_list = glob.glob(self._dataset_path + '/**/*.jpg', recursive=True)
        for filename in image_files_list[0:10]:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self._img_size, self._img_size))
            data = np.array(backend.normalize(image),dtype=np.float32)
            data = np.expand_dims(data, 0)
        yield [data]

    def convert_edgetpu(self,model_path):
        output_path = os.path.dirname(model_path)
        print(output_path)
        result = subprocess.run(["edgetpu_compiler", "--out_dir", output_path, model_path])
        print(result.returncode)

    def convert_k210(self,model_path,dataset_path):
        output_name = os.path.basename(model_path).split(".")[0]+".kmodel"
        output_path = os.path.join(os.path.dirname(model_path),output_name)
        print(output_path)
        result = subprocess.run([k210_converter_path, "compile",model_path,output_path,"-i","tflite","--dataset",dataset_path])
        print(result.returncode)

    def convert_tflite(self, model_path, model_layers, target=None):
        model = load_model(model_path,compile=False)
        model.save(model_path,overwrite=True,include_optimizer=False)
        yolo = 'reshape_1' in model_layers[-1].name
        if yolo and target=='k210': 
            print("Converting to tflite without Reshape layer for K210 Yolo")
            output_layer = model_layers[-2].name+'/BiasAdd'
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path,output_arrays=[output_layer])

        elif target == 'edgetpu':
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_gen
            #converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_quant_model = converter.convert()

        else:
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
        tflite_model = converter.convert()
        open(os.path.join (model_path.split(".")[0] + '.tflite'), "wb").write(tflite_model)

    def convert_model(self,model_path,model_layers,dataset_path=None):

        if 'k210' in self._converter_type:
            self.convert_tflite(model_path,model_layers, 'k210')
            self.convert_k210(model_path.split(".")[0] + '.tflite',dataset_path)

        if 'edgetpu' in self._converter_type:
            self.convert_tflite(model_path,model_layers, 'edgetpu')
            self.convert_edgetpu(model_path.split(".")[0] + '.tflite')

        if 'tflite' in self._converter_type:
            self.convert_tflite(model_path,model_layers)

    def save_frozen_graph(self,model, path, train_date):
        output_node_names = [node.op.name for node in model.outputs]
        print(output_node_names)
        input_node_names = [node.op.name for node in model.inputs]
        output_layer = model.layers[-1].name+'/BiasAdd'
        sess = K.get_session()
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
        graph_io.write_graph(constant_graph, "" , os.path.join (path, train_date + '.pb'), as_text=False)

    def convert_tensorrt(self):
        pass

