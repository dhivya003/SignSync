import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Set dataset path
base_path = "dataset/word"

# Image and batch size
img_size = (64, 64)  
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=30,  # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Added shear transformation
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation (no augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Load training data
train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Compute class weights for handling imbalance
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Load MobileNetV2 pre-trained model (without top layers)
base_model = keras.applications.MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights="imagenet")

# Unfreeze some layers of MobileNetV2 for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze initial 100 layers
    layer.trainable = False

# Create model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.6),  # Increased dropout to 0.6
    keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile model with a reduced learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Learning rate scheduler to reduce LR after plateau
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model with class weights
history = model.fit(train_generator, 
                    epochs=30, 
                    validation_data=val_generator, 
                    class_weight=class_weights_dict,  # Handling class imbalance
                    callbacks=[early_stopping, lr_scheduler])

# Save the trained model
model.save("models/mobilenet_word_model.h5")

# Evaluate the model
train_loss, train_accuracy = model.evaluate(train_generator)
print(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")

test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Model Loss")
plt.show()
