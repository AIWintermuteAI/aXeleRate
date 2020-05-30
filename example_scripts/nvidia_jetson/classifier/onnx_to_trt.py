import engine as eng
import argparse
import os
from onnx import ModelProto
import tensorrt as trt 
 
batch_size = 1 
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--onnx', help='File path to .onnx model', required=True)
parser.add_argument('--precision', help='FP32 or FP16', required=True)
args = parser.parse_args()

engine_name = os.path.dirname(args.onnx)+os.path.basename(args.onnx).split('.')[0]+'.plan'

model = ModelProto()
with open(args.onnx, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
shape = [batch_size , d0, d1 ,d2]
engine = eng.build_engine(args.onnx, shape = shape, precision=args.precision)
eng.save_engine(engine, engine_name) 
