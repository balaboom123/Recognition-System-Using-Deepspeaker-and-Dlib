import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
import dlib
import torch
from pathlib import Path
# from models.experimental import attempt_load
# from ultralytics.data.loaders import LoadImages
# from ultralytics.utils.ops import non_max_suppression

def face_feature(dir, interpolation=1, useless_img="save"):
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
	shape_predict_model = os.path.join(script_dir, "model\\shape_predictor_5_face_landmarks.dat")

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

			# covert to gray color
			# gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			# interpolation
			# scale_percent = interpolation
			# width = int(img.shape[1] * scale_percent / 100)
			# height = int(img.shape[0] * scale_percent / 100)
			# dim = (width, height)
			# resized_gray_face = cv2.resize(img, dim)

			# detect face
			detection = detector(img)
			print(person_img)
			print("it is detector", detection)

			# If training image contains not only one face, delete or skip it
			# If training image contains only one face, compute face description and save result
			if len(detection) != 1:
				if useless_img == "delete":
					print(f"Deleting {person}/{person_img} as it has {len(detection)} faces.")
					# os.remove(dir + person + "/" + person_img)
				else:
					print(f"Retain {person}/{person_img} as it has {len(detection)} faces.")

			else:
				left, top, right, bottom = detection[0].boxes.xywh[0].tolist()

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
data_dir = os.path.join(script_dir, "train_data")

vector, person = face_feature(data_dir, 80, "delete")
print("it is vector:", vector)
print("it is person:", person)

