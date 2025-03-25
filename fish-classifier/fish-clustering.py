import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Paths
image_dir = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\fish_on_static'

# Load pretrained CNN (VGG16), remove classification layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Prepare images and extract features
features = []
images = []
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = model.predict(img_array).flatten()
    features.append(feature)
    images.append(img_name)

features = np.array(features)

# Cluster with K-Means
num_clusters = 8  # Adjust based on how many species I think there are
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features)
labels = kmeans.labels_

# Organize images by cluster
for idx, label in enumerate(labels):
    cluster_dir = f'C:\\Users\\playi\\PycharmProjects\\fish-detecting\\fish-classifier\\class-data\\{label}'
    os.makedirs(cluster_dir, exist_ok=True)
    img_path = os.path.join(image_dir, images[idx])
    dst_path = os.path.join(cluster_dir, images[idx])
    os.rename(img_path, dst_path)

print("Images sorted into clusters!")
