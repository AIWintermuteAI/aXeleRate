import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="Segnet_best_val_loss.tflite")
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()
for op in details:
    print(op)
