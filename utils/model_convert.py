import tensorflow.keras.backend as K

K.set_learning_phase(0)
import tensorflow as tf
import keras2onnx
from tensorflow.keras.models import load_model

MODEL_PATH = ""
model = load_model(MODEL_PATH)

onnx_model = keras2onnx.convert_keras(model, "pose_model")

with open("model.onnx", "wb") as file:
    file.write(onnx_model.SerializeToString())
