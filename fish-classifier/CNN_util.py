import os
import cv2
import numpy as np
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths and hyperparameters
DATASET_DIR = r'C:\Users\playi\PycharmProjects\fish-detecting\fish-classifier\class-data\round 4-8'
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 8
EPOCHS = 50  # Increased for better learning


def load_data(data_dir, batch_size=BATCH_SIZE, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Loads images using ImageDataGenerator for both training and testing."""

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # Normalization & splitting

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator


def preprocess_image(image):
    """Preprocesses a single image before feeding it into the model."""
    img = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    return img / 255.0


def build_model():
    """Creates a CNN model using MobileNetV2 as a base."""

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Reduce overfitting
    x = Dense(128, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')  # Output layer

    model = Model(inputs=base_model.input, outputs=x)

    # Freeze base model layers to keep pretrained features
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model(model, test_generator):
    """Evaluates the model on test data."""
    loss, accuracy = model.evaluate(test_generator)
    return accuracy
