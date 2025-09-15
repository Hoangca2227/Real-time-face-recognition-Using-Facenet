
import streamlit as st
import cv2
import os
import numpy as np
import pickle
import mtcnn
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
from architecture import InceptionResNetV2

# ====== Setup =======
face_data = 'Faces/'
encodings_path = 'encodings/encodings.pkl'
required_shape = (160, 160)
l2_normalizer = Normalizer('l2')

# Load FaceNet
face_encoder = InceptionResNetV2()
face_encoder.load_weights("facenet_keras_weights.h5")

# MTCNN detector
face_detector = mtcnn.MTCNN()

# Load encodings
if os.path.exists(encodings_path):
    with open(encodings_path, 'rb') as f:
        encoding_dict = pickle.load(f)
else:
    encoding_dict = {}


# ===== Helper =====
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def register_user(name):
    os.makedirs(os.path.join(face_data, name), exist_ok=True)

    cap = cv2.VideoCapture(0)
    st.write("Capturing images for user:", name)

    captured_encodes = []
    for i in range(10):  # chụp 10 ảnh
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = face_detector.detect_faces(img_rgb)
        if len(x) == 0:
            continue
        x1, y1, w, h = x[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        face = img_rgb[y1:y2, x1:x2]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        captured_encodes.append(encode)

    cap.release()

    if captured_encodes:
        encode = np.sum(captured_encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[name] = encode

        with open(encodings_path, 'wb') as f:
            pickle.dump(encoding_dict, f)

        st.success(f"User {name} registered successfully!")


def delete_user(name):
    if name in encoding_dict:
        encoding_dict.pop(name)
        with open(encodings_path, 'wb') as f:
            pickle.dump(encoding_dict, f)
        st.success(f"Deleted user {name}")
    else:
        st.error("User not found")


def start_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = face_detector.detect_faces(img_rgb)

        for res in x:
            x1, y1, w, h = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            face = img_rgb[y1:y2, x1:x2]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encode = l2_normalizer.transform(np.expand_dims(encode.reshape(1, -1)))[0]

            name = "Unknown"
            min_dist = 1e6
            for db_name, db_encode in encoding_dict.items():
                dist = np.linalg.norm(db_encode - encode)
                if dist < 0.7 and dist < min_dist:  # ngưỡng nhận diện
                    name = db_name
                    min_dist = dist

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


# ===== Streamlit UI =====
st.title("Face Recognition System with FaceNet")

col1, col2 = st.columns(2)
with col1:
    new_user = st.text_input("Enter User's Name")
    if st.button("Register User"):
        if new_user:
            register_user(new_user)
        else:
            st.warning("Please enter a name.")

with col2:
    del_user = st.text_input("Delete User by Name")
    if st.button("Delete User"):
        if del_user:
            delete_user(del_user)
        else:
            st.warning("Please enter a name.")

if st.button("Start Face Recognition"):
    start_recognition()
