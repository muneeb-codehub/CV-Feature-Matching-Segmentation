import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# Paths
images_dir = "images"
masks_dir = "masks"

# Load data
def load_data(images_dir, masks_dir, size=(128,128)):
    X, Y = [], []
    for f in os.listdir(images_dir):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = Image.open(os.path.join(images_dir, f)).convert('L').resize(size)  # Convert to grayscale
        mask = Image.open(os.path.join(masks_dir, f)).convert('L').resize(size)  # Convert to grayscale
        X.append(np.array(img)/255.0)
        Y.append(np.array(mask)/255.0)
    X = np.expand_dims(np.array(X), -1)  # Add channel dimension
    Y = np.expand_dims(np.array(Y), -1)  # Add channel dimension
    return X, Y

X, Y = load_data(images_dir, masks_dir)

# Simple U-Net model
def unet_model(input_size=(128,128,1)):
    inputs = layers.Input(input_size)
    
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)
    
    u1 = layers.UpSampling2D()(c4)
    u1 = layers.Concatenate()([u1, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet_model()
model.summary()

# Train (CPU)
model.fit(X, Y, batch_size=1, epochs=10)
