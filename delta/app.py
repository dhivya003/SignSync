import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, Response, request, url_for, jsonify
from werkzeug.utils import secure_filename
import time

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

IMG_SIZE = (128, 128)
MODEL_PATH = "models/sample_mobilenet_word_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
CAPTURE_INTERVAL = 2

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

latest_prediction = {"class": "None", "confidence": 0.0}

def load_model_and_indices():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}.")
    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model output units: {model.layers[-1].units}")
    print("Loading class indices...")
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    print(f"Class indices: {class_indices}")
    if model.layers[-1].units != len(class_indices):
        raise ValueError(f"Model output units ({model.layers[-1].units}) do not match number of classes ({len(class_indices)}).")
    return model, class_indices

def preprocess_frame(frame, img_size=IMG_SIZE):
    try:
        frame_resized = cv2.resize(frame, img_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        return frame_batch
    except Exception as e:
        raise Exception(f"Frame preprocessing failed: {str(e)}")

def predict_frame(frame, model, class_indices):
    try:
        processed_frame = preprocess_frame(frame, IMG_SIZE)
        predictions = model.predict(processed_frame, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        adjusted_idx = predicted_class_idx - 1
        if adjusted_idx < 0:
            adjusted_idx = 0
        confidence = predictions[0][predicted_class_idx]
        index_to_class = {v: k for k, v in class_indices.items()}
        if adjusted_idx not in index_to_class:
            raise ValueError(f"Adjusted class index {adjusted_idx} not found in class_indices: {index_to_class}")
        predicted_class = index_to_class[adjusted_idx]
        print(f"Raw predictions: {predictions[0]}")
        print(f"Original index: {predicted_class_idx}, Adjusted index: {adjusted_idx}, Class: {predicted_class}, Confidence: {confidence:.4f}")
        return predicted_class, confidence * 100
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}") from e
    
def generate_frames(model, class_indices):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    last_capture_time = time.time() - CAPTURE_INTERVAL
    global latest_prediction
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            try:
                predicted_class, confidence = predict_frame(frame, model, class_indices)
                latest_prediction = {"class": predicted_class, "confidence": confidence}
                last_capture_time = current_time
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                latest_prediction = {"class": "Error", "confidence": 0.0}
        label = f"Class: {latest_prediction['class']}, Confidence: {latest_prediction['confidence']:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(model, class_indices), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict_number', methods=['POST'])
def predict_number():
    if 'image' not in request.files:
        return render_template('upload.html', error="No image uploaded"), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('upload.html', error="No selected file"), 400
    filename = secure_filename(image_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(file_path)
    try:
        img = cv2.imread(file_path)
        if img is None:
            return render_template('upload.html', error="Invalid image file"), 400
        predicted_class, confidence = predict_frame(img, model, class_indices)
        return render_template('upload.html', 
                               prediction=predicted_class, 
                               confidence=confidence, 
                               image_url=url_for('static', filename=f'uploads/{filename}'))
    except Exception as e:
        return render_template('upload.html', error=str(e)), 500

@app.route('/display-sign', methods=['POST'])
def display_sign():
    data = request.get_json()
    spoken_text = data.get("text", "").lower()

  
    gif_path = os.path.join("static", "ISL_Gifs", f"{spoken_text}.gif")

    if os.path.exists(gif_path):
        return jsonify({"image": url_for('static', filename=f"ISL_Gifs/{spoken_text}.gif", _external=True)})

   
    words_signs = []
    words = spoken_text.split() 

    for word in words:
        letter_gifs = []
        for letter in word:
            letter_gif_path = os.path.join("static", "letters", f"{letter}.jpg")
            if os.path.exists(letter_gif_path):
                letter_gifs.append(url_for('static', filename=f"letters/{letter}.jpg", _external=True))

        if letter_gifs:
            words_signs.append({"word": word, "letters": letter_gifs})

    if words_signs:
        return jsonify({
            "words": words_signs,
            "delay": 1000  
        })

    return jsonify({"error": "No sign representation found"}), 404

@app.route('/speech-to-sign')
def speech_to_sign_page():
    return render_template('speech_to_sign.html')


@app.route('/prediction')
def get_prediction():
    global latest_prediction
    return jsonify(latest_prediction)


# Load model and class indices
model, class_indices = load_model_and_indices()

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)