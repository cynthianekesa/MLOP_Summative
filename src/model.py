from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
import pandas as pd
import numpy as np
from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Import any layers you need
from keras import layers

#Model
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3

input = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model1 = models.Sequential([
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape = input),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model1.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))