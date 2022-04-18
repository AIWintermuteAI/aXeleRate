import tensorflow as tf
import tensorflow.keras.backend as k
import subprocess
import os
import cv2
import argparse
import tarfile
import glob
import shutil
import numpy as np
import shlex

k210_converter_path=os.path.join(os.path.dirname(__file__),"ncc","ncc")
k210_converter_download_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'ncc_linux_x86_64.tar.xz')
nncase_download_url="https://github.com/kendryte/nncase/releases/download/v0.2.0-beta4/ncc_linux_x86_64.tar.xz"
cwd = os.path.dirname(os.path.realpath(__file__))

def run_command(cmd, cwd=None):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash', universal_newlines=True, cwd=cwd) as p:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line)    
        exit_code = p.poll()
    return exit_code

class Converter(object):
    def __init__(self, converter_type, backend=None, dataset_path=None):
        if 'tflite' in converter_type:
            print('Tflite Converter ready')

        if 'k210' in converter_type:
            if os.path.exists(k210_converter_path):
                print('K210 Converter ready')
            else:
                print('Downloading K210 Converter')
                _path = tf.keras.utils.get_file(k210_converter_download_path, nncase_download_url)     
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
                cmd = "bash install_edge_tpu_compiler.sh"
                result = run_command(cmd, cwd)
                print(result)    
                
        if 'onnx' in converter_type:
            try:
                import tf2onnx
            except:
                cmd = "pip install tf2onnx"
                result = run_command(cmd, cwd)
                print(result)              
                
        self._converter_type = converter_type
        self._backend = backend
        self._dataset_path=dataset_path

    def edgetpu_dataset_gen(self):
        num_imgs = 300
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
            yield [data]

    def k210_dataset_gen(self):
        num_imgs = 300
        image_files_list = []
        from axelerate.networks.common_utils.feature import create_feature_extractor
        backend = create_feature_extractor(self._backend, [self._img_size[0], self._img_size[1]])
        image_search = lambda ext : glob.glob(self._dataset_path + ext, recursive=True)
        for ext in ['/**/*.jpg', '/**/*.jpeg', '/**/*.png']: image_files_list.extend(image_search(ext))
        temp_folder = os.path.join(os.path.dirname(__file__),'tmp')
        os.mkdir(temp_folder)
        for filename in image_files_list[:num_imgs]:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self._img_size[0], self._img_size[1]))
            data = np.array(backend.normalize(image), dtype=np.float32)
            data = np.expand_dims(data, 0)
            bin_filename = os.path.basename(filename).split('.')[0]+'.bin'
            with open(os.path.join(temp_folder, bin_filename), "wb") as f: 
                data = np.transpose(data, [0, 3, 1, 2])
                data.tofile(f)
        return temp_folder

    def convert_edgetpu(self, model_path):
        output_path = os.path.dirname(model_path)
        print(output_path)
        cmd = "edgetpu_compiler --out_dir {} {}".format(output_path, model_path)
        print(cmd)
        result = run_command(cmd)
        print(result)

    def convert_k210(self, model_path):
        folder_name = self.k210_dataset_gen()
        output_name = os.path.basename(model_path).split(".")[0]+".kmodel"
        output_path = os.path.join(os.path.dirname(model_path),output_name)
        print(output_path)
        cmd = '{} compile "{}" "{}" -i tflite --weights-quantize-threshold 1000 --dataset-format raw --dataset "{}"'.format(k210_converter_path, model_path, output_path, folder_name)
        print(cmd)
        result = run_command(cmd)
        shutil.rmtree(folder_name, ignore_errors=True)
        print(result)

    def convert_onnx(self, model):
        import tf2onnx
        spec = (tf.TensorSpec((None, *self._img_size, 3), tf.float32, name="input"),)
        output_path = self.model_path.split(".")[0] + '.onnx'
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path = output_path)

    def convert_ncnn(self, model):
        spec = (tf.TensorSpec((None, *self._img_size, 3), tf.float32, name="input"),)
        output_path = self.model_path.split(".")[0] + '.onnx'
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path = output_path)

    def onnx_to_ncnn(self, input_shape, onnx="out/model.onnx", ncnn_param="out/conv0.param", ncnn_bin = "out/conv0.bin"):
        import os
        # onnx2ncnn tool compiled from ncnn/tools/onnx, and in the buld dir
        cmd = f"onnx2ncnn {onnx} {ncnn_param} {ncnn_bin}"       #可以更换工具目录
        os.system(cmd)
        with open(ncnn_param) as f:
            content = f.read().split("\n")
            if len(input_shape) == 1:
                content[2] += " 0={}".format(input_shape[0])
            else:
                content[2] += " 0={} 1={} 2={}".format(input_shape[2], input_shape[1], input_shape[0])
            content = "\n".join(content)
        with open(ncnn_param, "w") as f:
            f.write(content)

    def convert_tflite(self, model, model_layers, target=None):
        model_type = model.name
        model.summary()

        if target=='k210': 
            if model_type == 'yolo' or model_type == 'segnet':
                print("Converting to tflite without Reshape for K210 YOLO")
                if len(model.outputs) == 2:
                    output1 = model.get_layer(name="detection_layer_1").output
                    output2 = model.get_layer(name="detection_layer_2").output
                    model = tf.keras.Model(inputs=model.input, outputs=[output1, output2])
                else:
                    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
                    
            model.input.set_shape(1 + model.input.shape[1:])
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

        elif target == 'edgetpu':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.edgetpu_dataset_gen
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        elif target == 'tflite_dynamic':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif target == 'tflite_fullint':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]            
            converter.representative_dataset = self.edgetpu_dataset_gen
            
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()
        open(os.path.join (self.model_path.split(".")[0] + '.tflite'), "wb").write(tflite_model)

    def convert_model(self, model_path):
        k.clear_session()
        k.set_learning_phase(0)
        model = tf.keras.models.load_model(model_path, compile=False)
        model_layers = model.layers
        self._img_size = model.input_shape[1:3]
        self.model_path = os.path.abspath(model_path)

        if 'k210' in self._converter_type:
            self.convert_tflite(model, model_layers, 'k210')
            self.convert_k210(self.model_path.split(".")[0] + '.tflite')

        if 'edgetpu' in self._converter_type:
            self.convert_tflite(model, model_layers, 'edgetpu')
            self.convert_edgetpu(model_path.split(".")[0] + '.tflite')

        if 'onnx' in self._converter_type:
            self.convert_onnx(model)
            
        if 'ncnn' in self._converter_type:
            model.save(model_path.split(".")[0])
            self.convert_onnx(model_path, model_layers)
            self.onnx_to_ncnn(model_path)

        if 'tflite' in self._converter_type:
            self.convert_tflite(model, model_layers, self._converter_type)

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
