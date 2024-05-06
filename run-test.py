import cv2
import time
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import dlib
import json
from biometric_recognition.face_feature_extract import pca, landmark, dct
from biometric_recognition.face_feature_extract import face_detect_resize


# load model
script_dir = os.path.dirname(os.path.abspath(__file__))
person_root = f"{script_dir}/data/person_data"
person_root = f"{script_dir}/data/person_data"
recognizer_model = os.path.join(
    script_dir,
    "biometric_recognition/model/dlib_face_recognition_resnet_model_v1.dat")
sp5 = os.path.join(
    script_dir,
    "biometric_recognition/model/shape_predictor_5_face_landmarks.dat")
sp68 = os.path.join(
    script_dir,
    "biometric_recognition/model/shape_predictor_68_face_landmarks.dat")

# dlib
recognizer = dlib.face_recognition_model_v1(recognizer_model)
shape_predictor_5 = dlib.shape_predictor(sp5)
shape_predictor_68 = dlib.shape_predictor(sp68)
detector = dlib.get_frontal_face_detector()

# def dict
landmark_x = []
landmark_y = []

pca_x = []
pca_y = []

dct_x = []
dct_y = []

all_entries = os.listdir(person_root)
person_names = [
    entry for entry in all_entries if os.path.isdir(
        os.path.join(
            person_root,
            entry))]

for person_name in person_names[:13]:
    vectors = [[], [], []]
    person_name_path = os.path.join(person_root, person_name)
    person_json = f"{person_root}/{person_name}.json"

    with open(person_json, "r") as json_file:
        data_dict = json.load(json_file)

    landmark_y.extend([person_name])
    pca_y.extend([person_name])
    dct_y.extend([person_name])

    length = len(data_dict["landmark_feature_68"])
    for i in range(length):
        vectors[0].extend([data_dict["landmark_feature_68"][i]])
        vectors[1].extend([data_dict["pca_feature35"][i]])
        vectors[2].extend([data_dict["dct_feature"][i]])

    avg_vectors = np.mean(vectors[0], axis=0)
    landmark_x.extend([avg_vectors])

    avg_vectors = np.mean(vectors[1], axis=0)
    pca_x.extend([avg_vectors])

    avg_vectors = np.mean(vectors[2], axis=0)
    dct_x.extend([avg_vectors])


print("name include", landmark_y)
cap = cv2.VideoCapture(0)
print(cap.isOpened())

count = 0
while True:
    count += 1

    # Capture frame-by-frame
    ret, frame = cap.read()

    if count % 48 != 0:
        continue

    rect, _, resize_img, face_count = face_detect_resize(frame, detector)

    if face_count != 1:
        print("No face detected")
        continue

    # feature extraction
    landmark_feature_68=landmark(
        frame, rect, recognizer, shape_predictor_68, detector)
    pca_feature=pca(resize_img, n_components=40)
    dct_feature=dct(resize_img)

    euclidean_distance=[
        np.linalg.norm(
            landmark_feature_68 -
            avg_vector) for avg_vector in landmark_x]
    min_index=np.argmin(euclidean_distance)
    print(f"landmark: {landmark_y[min_index]}")

    euclidean_distance=[
        np.linalg.norm(
            pca_feature -
            avg_vector) for avg_vector in pca_x]
    min_index=np.argmin(euclidean_distance)
    print(f"pca: {pca_y[min_index]}")

    euclidean_distance=[
        np.linalg.norm(
            dct_feature -
            avg_vector) for avg_vector in dct_x]
    min_index=np.argmin(euclidean_distance)
    print(f"dct: {dct_y[min_index]}")

    if euclidean_distance[min_index] < 0.4:
        print(f"Prediction: {landmark_y[min_index]}")
        print("")

    else:
        print("")

    # Display the prediction on the frame
    # cv2.imshow('live', frame)
    # cv2.putText(frame, f"Prediction: {prediction}", (50, 50),
    # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture
cap.release()
cv2.destroyAllWindows()
