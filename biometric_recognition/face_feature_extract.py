import numpy as np
from sklearn.decomposition import PCA
import cv2
import dlib
from ultralytics import YOLO
import os
import face_recognition


def face_detect_resize(image: str, detector, img_size: int = 64):
	# If input is a string, assume it's a file path
	if isinstance(image, str):
		dlib_image = dlib.load_rgb_image(image)
	# If input is a NumPy array, assume it's an image
	elif isinstance(image, np.ndarray):
		dlib_image = image
	else:
		raise ValueError("Input must be either a file path or a NumPy array representing an image")

	detection = detector(dlib_image, 0)
	face_count = len(detection)
	if face_count != 1:
		return None, None, None, None

	# load img for opencv format
	rect = detection[0]
	left = max(rect.left(), 0)
	top = max(rect.top(), 0)
	right = max(rect.right(), 0)
	bottom = max(rect.bottom(), 0)

	cv2_image = cv2.cvtColor(dlib_image, cv2.COLOR_RGB2GRAY)
	cv2_image = cv2_image[int(top):int(bottom), int(left):int(right)]
	resize_image = cv2.resize(cv2_image, (img_size, img_size))

	return detection[0], dlib_image, resize_image, face_count


def landmark(
	face,
	rect,
	recognizer,
	shape_predictor,
	save_img: bool = False):

	# get the shape of the face
	face_shape = shape_predictor(face, rect)
	face_aligned = dlib.get_face_chip(face, face_shape, size=150, padding=0.25)

	face_descript = recognizer.compute_face_descriptor(face_aligned)
	face_descript = np.array(face_descript)

	# if save_img == True, save the image with the facial landmarks
	if save_img:
		img_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
		for part in face_shape.parts():
			# Draw a circle at each landmark
			cv2.circle(img_cv, (part.x, part.y), 3, (0, 255, 0), -1)

		cv2.imwrite("face_shape_output.jpg", img_cv)
		cv2.imwrite("face_aligned_output.jpg", face_aligned)

	return face_descript


def face_encoding(img_path):
	face = face_recognition.load_image_file(img_path)
	face_bounding_boxes = face_recognition.face_locations(face)

	# If training image contains exactly one face
	if len(face_bounding_boxes) == 1:
		face_enc = face_recognition.face_encodings(face)[0]
	else:
		# os.remove(img_path)
		return None
	return face_enc


def face_encoding_real_time(face):
	face_bounding_boxes = face_recognition.face_locations(face)

	# If training image contains exactly one face
	if len(face_bounding_boxes) == 1:
		face_enc = face_recognition.face_encodings(face)[0]
	else:
		# os.remove(img_path)
		return None
	return face_enc