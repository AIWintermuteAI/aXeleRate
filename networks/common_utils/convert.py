import tensorflow as tf
import os
import keras.backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def save_tflite(path, train_date):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(os.path.join (path, train_date + '.h5'))
    tflite_model = converter.convert()
    open(os.path.join (path, train_date + '.tflite'), "wb").write(tflite_model)

def save_frozen_graph(model, path, train_date):
    output_node_names = [node.op.name for node in model.outputs]
    print(output_node_names)
    input_node_names = [node.op.name for node in model.inputs]
    output_layer = model.layers[-1].name+'/BiasAdd'
    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
    graph_io.write_graph(constant_graph, "" , os.path.join (path, train_date + '.pb'), as_text=False)
    model.save("tmp.h5",include_optimizer=False)
    converter = tf.lite.TFLiteConverter.from_keras_model_file("tmp.h5",output_arrays=output_node_names)
    tflite_model = converter.convert()
    open(os.path.join (path, train_date + '.tflite'), "wb").write(tflite_model)

def convert_k210():
    pass

def convert_tensorrt():
    pass

def convert_edgetpu():
    pass
