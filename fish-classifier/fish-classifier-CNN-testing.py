import tensorflow as tf
import CNN_util

# Load test data
_, test_generator = CNN_util.load_data(CNN_util.DATASET_DIR)  # Using the validation set for testing

# Load the trained model
model = tf.keras.models.load_model("fish_species_classifier.h5")
print("Model loaded successfully.")

# Evaluate the model
accuracy = CNN_util.evaluate_model(model, test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
