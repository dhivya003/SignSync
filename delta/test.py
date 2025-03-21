import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "models/sample_mobilenet_word_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"

# Function to preprocess a single image
def preprocess_image(image_path, img_size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}. Check the file path or format.")
    img_resized = cv2.resize(img, img_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# Function to predict the class of an image
def predict_image(image_path, model, class_indices):
    try:
        processed_img = preprocess_image(image_path, IMG_SIZE)
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Reverse class_indices to map index to class name
        index_to_class = {v: k for k, v in class_indices.items()}
        if predicted_class_idx not in index_to_class:
            raise ValueError(f"Predicted class index {predicted_class_idx} not found in class_indices: {index_to_class}")
        
        predicted_class = index_to_class[predicted_class_idx]
        
        # Debugging: Print raw predictions
        print(f"Raw predictions: {predictions[0]}")
        print(f"Predicted index: {predicted_class_idx}, Class: {predicted_class}, Confidence: {confidence:.4f}")
        
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# Main function to test prediction
def test_image_prediction(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}. Please train the model first.")

    print("Loading trained model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    print("Loading class indices...")
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    print(f"Class indices: {class_indices}")

    # Verify model output matches class indices
    output_units = model.layers[-1].units
    if output_units != len(class_indices):
        raise ValueError(f"Model output units ({output_units}) do not match number of classes ({len(class_indices)}). Retrain the model.")

    print(f"Predicting class for image: {image_path}")
    predicted_class, confidence = predict_image(image_path, model, class_indices)

    print("\nPrediction Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    return predicted_class, confidence

# Execute the prediction
if __name__ == "__main__":
    # Your test image path
    image_path = "dataset/word/assistance/User_4/assistance_461_User4_461.jpg"

    print("Starting image prediction...")
    try:
        predicted_class, confidence = test_image_prediction(image_path)
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")