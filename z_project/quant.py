import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import pathlib


loaded_model = tf.keras.models.load_model('model_VGG19.h5')
loaded_model.summary()


#Convert to a tensorflow LITE MODEL
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()


#wrtie it out to a .tflite file
tflite_models_dir = pathlib.Path("./tflite/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"vgg19_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"vgg19_model_quant_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)


#Load the model into the interpreters
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()


train_image, test_image, train_label , test_label = np.load('8thtrial.npy', allow_pickle=True)

# Normalize the input image so that each pixel value is between 0 to 1.
train_image = train_image / 255.0
test_image = test_image / 255.0

#test the models on one image
test_image = np.expand_dims(test_image[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

import matplotlib.pylab as plt

plt.imshow(test_image[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_label[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)

test_image = np.expand_dims(test_image[0], axis=0).astype(np.float32)

input_index = interpreter_fp16.get_input_details()[0]["index"]
output_index = interpreter_fp16.get_output_details()[0]["index"]

interpreter_fp16.set_tensor(input_index, test_image)
interpreter_fp16.invoke()
predictions = interpreter_fp16.get_tensor(output_index)


plt.imshow(test_image[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_label[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)

#Evaluate the models
# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_ima in test_image:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_ima = np.expand_dims(test_ima, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_ima)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_label[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

print(evaluate_model(interpreter))
