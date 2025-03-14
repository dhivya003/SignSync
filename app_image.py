import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
import pyttsx3
import threading

app = Flask(__name__)

engine = pyttsx3.init()

engine.setProperty('rate', 150)  
engine.setProperty('volume', 1.0)  

model = tf.keras.models.load_model('models/image_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

IMG_SIZE = 64
CHANNELS = 3

labels_path = "dataset/Sign_language_data/train/labels"

class_labels = sorted(os.listdir(labels_path))

detected_sign = ""  
last_spoken_sign = ""  
def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    image = np.array(image, dtype=np.float32) / 255.0  
    
    if image.shape[2] != 3:
        raise ValueError("Input image should have 3 channels (RGB).")
    
    return image.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def speak_sign(sign):
    engine.say(sign)
    engine.runAndWait()

def predict_sign(frame, hand_landmarks):
    hand_img = preprocess_image(frame)
    
    prediction = model.predict(hand_img)
    
    print("Raw Prediction:", prediction)
    
    sign_label = np.argmax(prediction)
    
    print("Predicted label index:", sign_label)
    print("Predicted sign:", class_labels[sign_label])
    
    return class_labels[sign_label]

def generate_frames():
    global detected_sign, last_spoken_sign
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                def background_predict():
                    global detected_sign, last_spoken_sign
                    detected_sign = predict_sign(frame, hand_landmarks)
                    if detected_sign != last_spoken_sign:
                        threading.Thread(target=speak_sign, args=(detected_sign,)).start()
                        last_spoken_sign = detected_sign

                threading.Thread(target=background_predict).start()

                cv2.putText(frame, f'Sign: {detected_sign}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign_output')
def sign_output():
    return jsonify({"sign": detected_sign})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global detected_sign, last_spoken_sign
    detected_sign = ""  
    last_spoken_sign = ""  
    return jsonify({"status": "Camera Started", "sign": detected_sign})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global detected_sign, last_spoken_sign
    detected_sign = ""  
    last_spoken_sign = ""  
    return jsonify({"status": "Camera Stopped", "sign": detected_sign})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']
    
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    print("Original image shape:", image.shape) 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Converted image shape:", image.shape)  

    hand_landmarks = hands.process(image)

    if hand_landmarks.multi_hand_landmarks:
        hand_landmarks = hand_landmarks.multi_hand_landmarks[0]
        print("Hand landmarks detected:", hand_landmarks)  
        predicted_sign = predict_sign(image, hand_landmarks)
        print("Predicted sign:", predicted_sign)  
        return jsonify({"sign": predicted_sign})
    
    return jsonify({"error": "No sign detected in the image"})

if __name__ == '__main__':
    app.run(debug=True)
