import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("fish_species_classifier.h5")

# Load and preprocess image
img_path = 'path/to/new/fish_image.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
class_indices = train_generator.class_indices
label_map = {v: k for k, v in class_indices.items()}

print(f"Predicted species: {label_map[predicted_class]}")
