import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
BASE_PATH = "dataset/sample_word"  # Path to your dataset
IMG_SIZE = (128, 128)       # Image size for MobileNetV2
BATCH_SIZE = 32             # Batch size for training
MODEL_PATH = "models/sample_mobilenet_word_model.h5"  # Where to save the model
CLASS_INDICES_PATH = "models/class_indices.json"  # Where to save class indices

# Step 1: Preprocess the Dataset
def prepare_data(base_path, img_size, batch_size):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # No augmentation for validation and test, only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load all data
    all_generator = val_test_datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Get class indices
    class_indices = all_generator.class_indices

    # Collect file paths and labels
    all_files = all_generator.filepaths
    all_labels = all_generator.labels

    # Split into train (70%), validation (15%), and test (15%)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, train_size=0.7, stratify=all_labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, train_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create DataFrames
    train_df = pd.DataFrame({'filename': train_files, 'class': [str(label) for label in train_labels]})
    val_df = pd.DataFrame({'filename': val_files, 'class': [str(label) for label in val_labels]})
    test_df = pd.DataFrame({'filename': test_files, 'class': [str(label) for label in test_labels]})

    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator, class_indices

# Step 2: Train and Evaluate the Model
def train_model(base_path, img_size, batch_size):
    # Load and prepare data
    train_generator, val_generator, test_generator, class_indices = prepare_data(base_path, img_size, batch_size)

    # Verify data
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    if train_generator.samples == 0 or val_generator.samples == 0 or test_generator.samples == 0:
        raise ValueError("One or more data generators have no samples. Check your dataset.")

    # Load MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Build the model
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(len(class_indices), activation='softmax')
    ])

    # Compile the model
    lr_scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        alpha=0.00001
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stopping, tensorboard_callback]
    )

    # Save the model and class indices
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_indices, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Class indices saved to {CLASS_INDICES_PATH}")

    # Evaluate on all sets
    print("\nEvaluating model performance:")
    train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    # Print accuracies and losses
    print(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    # Generate predictions for confusion matrix
    print("\nGenerating predictions for confusion matrix...")
    test_generator.reset()  # Reset generator to start from beginning
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    class_names = list(class_indices.keys())

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Model Accuracy During Training")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Model Loss During Training")
    plt.show()

    return model, class_indices

# Main execution
if __name__ == "__main__":
    print("Starting preprocessing and training...")
    model, class_indices = train_model(BASE_PATH, IMG_SIZE, BATCH_SIZE)
    print("Training completed successfully!")