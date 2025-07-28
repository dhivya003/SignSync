import os
import pickle
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, url_for, jsonify
from werkzeug.utils import secure_filename
import time
import mediapipe as mp

app = Flask(__name__)

CAPTURE_INTERVAL = 2
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

latest_prediction = {"class": "", "confidence": 0.0}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
        26: 'Hello',
        27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
        32: 'You are welcome.',33:'My',34:'You',35:'He',36:'She',37:'It',38:'Eat',39:'Drink',40:'Go',41:'Come',42:'Wait',
        43:'Tea',44:'Stop',45:'Wait',46:'Telephone',47:'Name',48:'Money',49:'Walk',50:'Think'
    }

def load_model():
    """Load the trained model from pickle file"""
    #model_path = "signmodel.p"
    model_path = os.path.join(os.path.dirname(__file__), "signmodel.p")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
   
    #if not os.path.exists(model_path):
        #raise FileNotFoundError(f"Model file not found at {model_path}.")
    print("Loading trained model...")
    try:
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def extract_hand_landmarks(frame):
    """Extract normalized hand landmarks using MediaPipe"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            return np.asarray(data_aux)
    return None

def predict_frame(frame, model):
    """Predict the class of the frame using the trained model"""
    try:
        landmarks = extract_hand_landmarks(frame)
        if landmarks is None:
            return None, 0.0  
        landmarks = np.asarray([landmarks])
        prediction = model.predict(landmarks)
        prediction_proba = model.predict_proba(landmarks)
        predicted_class_idx = int(prediction[0])
        confidence = max(prediction_proba[0]) * 100  
        if predicted_class_idx not in labels_dict:
            raise ValueError(f"Predicted class index {predicted_class_idx} not found in labels_dict.")
        predicted_class = labels_dict[predicted_class_idx]
        print(f"Predicted class: {predicted_class}")  
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}") from e

def generate_frames(model):
    """Generate video frames with predictions only when hand is detected"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    last_capture_time = time.time() - CAPTURE_INTERVAL
    global latest_prediction
    latest_prediction = {"class": "", "confidence": 0.0}
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        frame = cv2.flip(frame, 1)  
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            try:
                predicted_class, confidence = predict_frame(frame, model)
                if predicted_class is not None:
                    latest_prediction = {"class": predicted_class, "confidence": confidence}
                else:
                    latest_prediction = {"class": "", "confidence": 0.0}
                last_capture_time = current_time
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                latest_prediction = {"class": "Error", "confidence": 0.0}
        if latest_prediction["class"]:
            label = f"Class: {latest_prediction['class']}"  
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
    return Response(generate_frames(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam1.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict_number', methods=['POST'])
def predict_number():
    """Handle image upload and prediction"""
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
        predicted_class, confidence = predict_frame(img, model)
        return render_template('upload.html',
                               prediction=predicted_class if predicted_class else "No hand detected",
                               confidence=confidence,  
                               image_url=url_for('static', filename=f'uploads/{filename}'))
    except Exception as e:
        return render_template('upload.html', error=str(e)), 500

@app.route('/display-sign', methods=['POST'])
def display_sign():
    """Convert spoken text to sign language representations"""
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
        return jsonify({"words": words_signs, "delay": 1000})
    return jsonify({"error": "No sign representation found"}), 404

@app.route('/speech-to-sign')
def speech_to_sign_page():
    return render_template('speech_to_sign.html')

@app.route('/prediction')
def get_prediction():
    global latest_prediction
    return jsonify(latest_prediction)

@app.route('/learn_signs')
def learn_signs():
    return render_template('learn_signs.html')


model = load_model()

if __name__ == "__main__":
    print("Starting Flask app...")
    #app.run(debug=True, host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)