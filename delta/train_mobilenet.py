import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset'

data = []
labels = []

if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as f:
        existing_data = pickle.load(f)
        data = existing_data['data']
        labels = existing_data['labels']

for dir_ in os.listdir(DATA_DIR):
    if str(dir_) in labels:
        continue  
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    x_.append(x)
                    y_.append(y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print('Hand landmarks extracted and saved!')


data = np.array(data)
labels = np.array(labels)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)



input_shape = (len(X_train[0]),)

base_model = MobileNetV2(
    include_top=False, 
    input_shape=(224, 224, 3), 
    weights='imagenet'
)


model = Sequential([
    Dense(512, activation='relu', input_shape=input_shape),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(labels)), activation='softmax')  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


with open('signmodel.p', 'wb') as f:
    pickle.dump(model, f)

print(' Model trained and saved as signmodel.p')
