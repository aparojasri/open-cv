import tensorflow as tf

# Load your Keras model (HDF5 or SavedModel)
model = tf.keras.models.load_model(r".\models\fer_emotion_cnn.keras")
 # or .h5

# Create converter from keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) set optimization flags — here none for pure float32
# converter.optimizations = []

tflite_model = converter.convert()

# Write to file
with open("models/emotion_fer.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved emotion_fer.tflite")
