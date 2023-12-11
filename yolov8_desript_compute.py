import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
import dlib
import torch
from pathlib import Path
import pprint

def facial_feature_extract(dir):
	"""
	return:
	encodings
	names
	"""
	# feature container
	encodings = []
	names = []

	# root dir
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# load model
	detector_model = os.path.join(script_dir, "model\\yolov8n.pt")
	recognizer_model = os.path.join(script_dir, "model\\dlib_face_recognition_resnet_model_v1.dat")
	shape_predict_model = os.path.join(script_dir, "model\\shape_predictor_68_face_landmarks.dat")

	detector = YOLO(detector_model)
	recognizer = dlib.face_recognition_model_v1(recognizer_model)
	shape_predictor = dlib.shape_predictor(shape_predict_model)

	# Training directory
	if dir[-1] != '/':
		dir += '/'
	train_dir = os.listdir(dir)

	# Loop through each person in the training directory
	for person in train_dir:
		pix = os.listdir(dir + person)

		# Loop through each training image for the current person
		for person_img in pix:
			# load img to dlib and opencv
			face = dlib.load_rgb_image(dir + person + "/" + person_img)
			img = dir + person + "/" + person_img

			# detect face
			detection = detector(img)
			print(person_img)

			left, top, right, bottom = detection[0].boxes.xywh[0].tolist()
			print(int(left), int(top), int(right), int(bottom))

			dlibRect = dlib.rectangle(int(left), int(top), int(right), int(bottom))

			# get the shape of the face
			face_shape = shape_predictor(face, dlibRect)
			face_aligned = dlib.get_face_chip(face, face_shape, size=150, padding=0.25)

			# get the facial feature extraction
			face_descript = recognizer.compute_face_descriptor(face_aligned)
			face_descript = np.array(face_descript)
			encodings.append(face_descript)
			names.append(person)

	return encodings, names

# train data dir
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data/train_data/image")

vector, person = facial_feature_extract(data_dir)
pp = pprint.PrettyPrinter(indent=1, compact=True, depth=1)
pp.pprint(vector)
pp.pprint(person)

