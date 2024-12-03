from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
import pandas as pd
import numpy as np

#Model
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3

input = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model1 = models.Sequential([
    scaling,
    data_augmentation,
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
model1.build(input_shape=input)
class WastePredictor:
    def __init__(self, train_dir, test_dir, model_dir):
        """
        Initializes a new instance of the LoanDefaultPredictor class.

        Args:
            train_dir (str): The directory containing the training datasets.
            test_dir (str): The directory containing the testing datasets.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_dir = model_dir
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
def plot_confusion_matrix(self):
        """
        Plot the confusion matrix using seaborn.

        Returns:
            None
        """
        matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()

def plot_training_history(self):
        '''
        Plot the training history of the model.

        Returns:
            None
        '''
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.savefig('training_history.png')
        plt.close()