from pyexpat import model
from keras.models import load_model
import tensorflow as tf

model=load_model("boyorgirl.h5")

converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

print("Model converted")


# Save the model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)