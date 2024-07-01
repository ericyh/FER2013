import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/eric/Documents/VS_CODE/FER2013/fer2013.csv")
df.columns = ["emotion", "pixels"]
emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

pixels = [np.array([float(j) / 255 for j in i.split(" ")]).reshape(48, 48) for i in df["pixels"].values]
emotions = tf.keras.utils.to_categorical(df["emotion"].values, 7)
x_train, x_val, x_test = np.split(pixels, [int(0.7*len(pixels)), int(0.85*len(pixels))]) #separate into training, validation and testing sets
y_train, y_val, y_test = np.split(emotions, [int(0.7*len(emotions)), int(0.85*len(emotions))]) #separate into training, validation and testing sets

print(x_train.shape)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(7, activation='softmax')
    ]
)

model.summary()
#Compile and train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))