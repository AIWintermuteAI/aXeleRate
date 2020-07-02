import tensorflow as tf
import keras
import subprocess
import os
import cv2
import argparse
import tarfile
import glob
import shutil
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

k210_converter_path=os.path.join(os.path.dirname(__file__),"ncc","ncc")
k210_converter_download_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'ncc_linux_x86_64.tar.xz')
nncase_download_url="https://github.com/kendryte/nncase/releases/download/v0.2.0-beta2/ncc_linux_x86_64.tar.xz"
cwd = os.path.dirname(os.path.realpath(__file__))

class Converter(object):
    def __init__(self, converter_type, backend=None, dataset_path=None):
        if 'tflite' in converter_type:
            print('Tflite Converter ready')

        if 'k210' in converter_type:
            if os.path.exists(k210_converter_path):
                print('K210 Converter ready')
            else:
                print('Downloading K210 Converter')
                _path = keras.utils.get_file(k210_converter_download_path, nncase_download_url)     
                print(_path)    
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

    def edgetpu_dataset_gen(self):
        num_imgs = None
        image_files_list = []
        from axelerate.networks.common_utils.feature import create_feature_extractor
        backend = create_feature_extractor(self._backend, [self._img_size[0], self._img_size[1]])
        image_search = lambda ext : glob.glob(self._dataset_path + ext, recursive=True)
        for ext in ['/**/*.jpg', '/**/*.jpeg', '/**/*.png']: image_files_list.extend(image_search(ext))

        for filename in image_files_list[:num_imgs]:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self._img_size[0], self._img_size[1]))
            data = np.array(backend.normalize(image), dtype=np.float32)
            data = np.expand_dims(data, 0)
            yield [out]

    def k210_dataset_gen(self):
        num_imgs = None
        image_files_list = []
        from axelerate.networks.common_utils.feature import create_feature_extractor
        backend = create_feature_extractor(self._backend, [self._img_size[0], self._img_size[1]])
        image_search = lambda ext : glob.glob(self._dataset_path + ext, recursive=True)
        for ext in ['/**/*.jpg', '/**/*.jpeg', '/**/*.png']: image_files_list.extend(image_search(ext))
        temp_folder = os.path.join(os.path.dirname(__file__),'tmp')
        os.mkdir(temp_folder)
        print(image_files_list)
        for filename in image_files_list[:num_imgs]:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self._img_size[0], self._img_size[1]))
            data = np.array(backend.normalize(image), dtype=np.float32)
            data = np.expand_dims(data, 0)
            bin_filename = os.path.basename(filename).split('.')[0]+'.bin'
            with open(os.path.join(temp_folder, bin_filename), "ba+") as f:
                data = np.transpose(data, [0, 3, 1, 2])
                data.tofile(f)
        return temp_folder

    def convert_edgetpu(self, model_path):
        output_path = os.path.dirname(model_path)
        print(output_path)
        result = subprocess.run(["edgetpu_compiler", "--out_dir", output_path, model_path])
        print(result.returncode)

    def convert_k210(self, model_path):
        folder_name = self.k210_dataset_gen()
        output_name = os.path.basename(model_path).split(".")[0]+".kmodel"
        output_path = os.path.join(os.path.dirname(model_path),output_name)
        print(output_path)
        result = subprocess.run([k210_converter_path, "compile", model_path,output_path,"-i","tflite", "--dataset-format", "raw", "--dataset", folder_name])
        shutil.rmtree(folder_name, ignore_errors=True)
        print(result.returncode)

    def convert_onnx(self, model_path, model_layers):
        import keras.backend as k
        k.clear_session()
        k.set_learning_phase(0)

        model = keras.models.load_model(model_path, compile=False)
        input_node_names = model.layers[0].get_output_at(0).name.split(':')[0]
        output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]
        sess = k.get_session()

        # The TensorFlow freeze_graph expects a comma-separated string of output node names.
        input_node_names_onnx= [model.layers[0].get_output_at(0).name]
        output_node_names_onnx = [model.layers[-1].get_output_at(0).name]
        print(output_node_names_onnx)
        print(input_node_names_onnx)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

        tf.reset_default_graph()
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(frozen_graph_def, name='')

            onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names=input_node_names_onnx, output_names=output_node_names_onnx)
            model_proto = onnx_graph.make_model("model")
            with open(model_path.split(".")[0] + '.onnx', "wb") as f:
               f.write(model_proto.SerializeToString())
        #sess.close()

    def convert_tflite(self, model_path, model_layers, target=None):
        yolo = 'reshape_1' in model_layers[-1].name
        if yolo and target=='k210': 
            print("Converting to tflite without Reshape layer for K210 Yolo")
            output_layer = model_layers[-2].name+'/BiasAdd'
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path, output_arrays=[output_layer])

        elif target == 'edgetpu':
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.edgetpu_dataset_gen
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_quant_model = converter.convert()

        else:
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
        tflite_model = converter.convert()
        open(os.path.join (model_path.split(".")[0] + '.tflite'), "wb").write(tflite_model)

    def convert_model(self, model_path):
        model = keras.models.load_model(model_path, compile=False)
        model_layers = model.layers
        self._img_size = model.inputs[0].shape[1:3]
        model.save(model_path, overwrite=True, include_optimizer=False)

        if 'k210' in self._converter_type:
            self.convert_tflite(model_path, model_layers, 'k210')
            self.convert_k210(model_path.split(".")[0] + '.tflite')

        if 'edgetpu' in self._converter_type:
            self.convert_tflite(model_path,model_layers, 'edgetpu')
            self.convert_edgetpu(model_path.split(".")[0] + '.tflite')

        if 'onnx' in self._converter_type:
            import tf2onnx
            self.convert_onnx(model_path, model_layers)

        if 'tflite' in self._converter_type:
            self.convert_tflite(model_path,model_layers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Keras model conversion to .kmodel, .tflite, or .onnx")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="path to keras model")
    parser.add_argument("--converter_type", type=str, default='k210',
                        help="batch size")
    parser.add_argument("--dataset_path", type=str, required=False,
                        help="path to calibration dataset")
    parser.add_argument("--backend", type=str, default='MobileNet7_5',
                    help="network feature extractor, e.g. Mobilenet/YOLO/NASNet/etc")                    
    args = parser.parse_args()
    converter = Converter(args.converter_type, args.backend, args.dataset_path)
    converter.convert_model(args.model_path)
