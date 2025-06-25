import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model_path = 'models/final_model.h5'
tflite_path = 'models/final_model.tflite'

try:
    model = load_model(model_path, compile=False)
    tf.saved_model.save(model, model_path.replace('.h5', ''))
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path.replace('.h5', ''))
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
except Exception as e:
    print(f"Error: {e}")
    model = load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
