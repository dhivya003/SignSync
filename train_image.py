import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

train_image_dir = 'dataset/Sign_language_data/train/images'
train_label_dir = 'dataset/Sign_language_data/train/labels'
test_image_dir = 'dataset/Sign_language_data/test/images'
test_label_dir = 'dataset/Sign_language_data/test/labels'

IMG_SIZE = (64, 64)  
num_classes = 8
class_names = ['0','1','Hello', 'IloveYou', 'No', 'Please', 'Thanks', 'Yes']

def load_images(image_dir, label_dir, IMG_SIZE):
    images = []
    labels = []

    for class_id, class_name in enumerate(class_names):
        class_image_dir = os.path.join(image_dir, class_name)
        class_label_dir = os.path.join(label_dir, class_name)

        for image_name in os.listdir(class_image_dir):
            image_path = os.path.join(class_image_dir, image_name)
            label_path = os.path.join(class_label_dir, image_name)

            img = cv2.imread(image_path)
            img = cv2.resize(img, IMG_SIZE)  
            
            if len(img.shape) == 2:  
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            images.append(img)
            
            label = np.zeros(num_classes)
            label[class_id] = 1
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

X_train, y_train = load_images(train_image_dir, train_label_dir, IMG_SIZE)
X_val, y_val = load_images(test_image_dir, test_label_dir, IMG_SIZE)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")

X_train = X_train / 255.0
X_val = X_val / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/image_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

