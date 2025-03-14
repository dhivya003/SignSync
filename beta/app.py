import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template,url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)


IMG_SIZE = (64, 64) 
dataset_path = "dataset/word"

class_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

model = load_model('models/indian_word_model.h5')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(image_path, IMG_SIZE):
    img = cv2.imread(image_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_equalized = cv2.equalizeHist(img_gray)

    img = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, IMG_SIZE)

   
    img = np.array(img, dtype=np.float32) / 255.0  

    return img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_number', methods=['POST'])
def predict_number():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_file.filename))
    image_file.save(file_path)

    try:
        test_img = preprocess_image(file_path, IMG_SIZE)
        prediction = model.predict(test_img)
        predicted_class_id = np.argmax(prediction)
        predicted_class = class_names[predicted_class_id]
        confidence = float(np.max(prediction))  

        return jsonify({"predicted_number": predicted_class, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/upload')
def upload():
    return render_template('upload.html')

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



if __name__ == "__main__":
    app.run(debug=True) 