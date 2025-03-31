import CNN_util
from tensorflow.keras.callbacks import EarlyStopping

# Load training & validation data
train_generator, val_generator = CNN_util.load_data(CNN_util.DATASET_DIR)

# Build the CNN model
model = CNN_util.build_model()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=CNN_util.EPOCHS, callbacks=[early_stopping], verbose=1)

# Save the trained model
model.save("fish_species_classifier.h5")
print("Model training complete and saved.")
