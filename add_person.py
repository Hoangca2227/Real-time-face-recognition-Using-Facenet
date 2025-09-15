import os
import cv2
import pickle
import numpy as np
import mtcnn
from sklearn.preprocessing import Normalizer
from architecture import InceptionResNetV2
from tensorflow.keras.models import load_model

# Config
required_shape = (160,160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights("facenet_keras_weights.h5")
face_detector = mtcnn.MTCNN()
l2_normalizer = Normalizer('l2')
encoding_path = "encodings/encodings.pkl"

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_embedding(person_dir):
    encodes = []
    for image_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, image_name)
        img_BGR = cv2.imread(img_path)
        if img_BGR is None: 
            continue
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        x = face_detector.detect_faces(img_RGB)
        if len(x) == 0: 
            continue
        x1, y1, w, h = x[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1+w, y1+h
        face = img_RGB[y1:y2, x1:x2]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face = np.expand_dims(face, axis=0)
        encodes.append(face_encoder.predict(face)[0])
    if encodes:
        encode = np.sum(encodes, axis=0)
        return l2_normalizer.transform([encode])[0]
    return None

def add_new_person(name, person_dir):
    # Load dict cũ
    if os.path.exists(encoding_path):
        with open(encoding_path, "rb") as f:
            encoding_dict = pickle.load(f)
    else:
        encoding_dict = {}

    encode = get_embedding(person_dir)
    if encode is not None:
        encoding_dict[name] = encode
        with open(encoding_path, "wb") as f:
            pickle.dump(encoding_dict, f)
        print(f"✅ Added {name} to encodings.pkl")
    else:
        print("⚠️ No face detected for", name)

# Example:
# add_new_person("Hoang", "Faces/Hoang")
