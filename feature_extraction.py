import time

import librosa
import numpy as np
import os
import json
import dlib

from biometric_recognition.voice_feature_extract import mfcc, lpc, lpcc
from biometric_recognition.face_feature_extract import pca, dct, landmark
from biometric_recognition.face_feature_extract import face_detect_resize, face_encoding
from biometric_recognition.feature_process import retain_avg_vector
from biometric_recognition.feature_save import feature_save, check_reapetition
"""
# voice feature extraction
voice_path = "/content/Voice_S1252001.wav"

# sample_rate, signal = wavfile.read(voice_path)
signal, sr = librosa.load(voice_path, sr=None)

mfccs_feature13 = mfcc(signal, sr, n_mfcc=13)
mfccs_feature39 = mfcc(signal, sr, n_mfcc=39)
print(f"mfccs is: {mfccs_feature39}")

lpc_feature16 = lpc(signal, order=16)
lpc_feature32 = lpc(signal, order=32)
lpc_feature44 = lpc(signal, order=44)
lpc_feature48 = lpc(signal, order=48)
print(f"lpc is: {lpc_feature48}")

lpcc_feature16 = lpcc(lpc_feature16, lpcc_order=16, lpc_order=16)
lpcc_feature32 = lpcc(lpc_feature32, lpcc_order=32, lpc_order=32)
lpcc_feature44 = lpcc(lpc_feature44, lpcc_order=44, lpc_order=44)
lpcc_feature48 = lpcc(lpc_feature48, lpcc_order=48, lpc_order=48)
print(f"lpcc is: {lpcc_feature48}")
"""

# face feature extraction
# root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
person_root = f"{script_dir}/data/person_data"
# dir_paths = [os.path.join(person_root, p) for p in os.listdir(person_root)]
# img_paths = [[os.path.join(p, x) for x in os.listdir(p) if x.endswith(".jpg")] for p in dir_paths]

# load model
# detector_model = os.path.join(script_dir, "biometric_recognition/model/detect8l.pt")
recognizer_model = os.path.join(script_dir, "biometric_recognition/model/dlib_face_recognition_resnet_model_v1.dat")
sp5 = os.path.join(script_dir, "biometric_recognition/model/shape_predictor_5_face_landmarks.dat")
sp68 = os.path.join(script_dir, "biometric_recognition/model/shape_predictor_68_face_landmarks.dat")

# dlib
recognizer = dlib.face_recognition_model_v1(recognizer_model)
shape_predictor_5 = dlib.shape_predictor(sp5)
shape_predictor_68 = dlib.shape_predictor(sp68)
detector = dlib.get_frontal_face_detector()

all_entries = os.listdir(person_root)
person_names = [entry for entry in all_entries if os.path.isdir(os.path.join(person_root, entry))]
for person_name in person_names[:13]:
	# def dict
	feature_dict = {
		"file_name": [],
		"landmark_feature_5": [],
		"landmark_feature_68": [],
		"pca_feature35": [],
		"dct_feature": []
	}

	person_name_path = os.path.join(person_root, person_name)
	
	person_json = f"{person_root}/{person_name}.json"
	if os.path.exists(person_json):
		with open(person_json, "r") as json_file:
			json_dict = json.load(json_file)
	else:
		json_dict = {"file_name": []}

	for img_name in os.listdir(person_name_path):
		check = check_reapetition(json_dict, img_name)
		if not check:
			print(f"{img_name} is already in the feature.json")
			continue

		img_path = os.path.join(person_name_path, img_name)
		print(f"START {img_name} feature extraction.")
		rect, img, resize_img, face_count = face_detect_resize(img_path, detector)
		if face_count != 1:
			continue

		feature_dict["file_name"].append(img_name)

		# landmark_feature_5 = landmark(img, rect, recognizer, shape_predictor_5, detector)
		# feature_dict["landmark_feature_5"].append(list(landmark_feature_5))

		landmark_feature_68 = landmark(img, rect, recognizer, shape_predictor_68, detector)
		feature_dict["landmark_feature_68"].append(list(landmark_feature_68))

		pca_feature35 = pca(resize_img, n_components=40)
		feature_dict["pca_feature35"].append(list(pca_feature35))

		dct_feature = dct(resize_img)
		feature_dict["dct_feature"].append(list(dct_feature))

		# print(f"Finish {img_name} feature extraction.")

	vectors = feature_dict["landmark_feature_68"]
	vectors = retain_avg_vector(vectors)
	feature_dict["landmark_feature_68"] = vectors

	if len(feature_dict["file_name"]) > 0:
		feature_save(person_json, feature_dict)


