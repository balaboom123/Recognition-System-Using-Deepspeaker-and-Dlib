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


def yolo_detect_resize(image_path: str, yolo_detect: str, img_size: int = 128):
	# detect face
	detector = YOLO(yolo_detect)
	results = detector(image_path)

	# get the face class count
	names = detector.names
	face_id = list(names)[list(names.values()).index('face')]
	face_count = results[0].boxes.cls.tolist().count(face_id)
	if face_count != 1:
		os.remove(image_path)
		return None, None, None

	# load img for opencv format
	xyxy = results[0].boxes.xyxy[0].tolist()
	left, top, right, bottom = xyxy
	cv2_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	cv2_image = cv2_image[int(top):int(bottom), int(left):int(right)]
	cv2_image = cv2.resize(cv2_image, (img_size, img_size))

	return xyxy, cv2_image, face_count


def pca(image, n_components=0.95, resize: int = 64):
	# If input is a string, assume it's a file path
	if isinstance(image, str):
		image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (resize, resize))
	# If input is a NumPy array, assume it's an image
	elif isinstance(image, np.ndarray):
		image = image
	else:
		raise ValueError("Input must be either a file path or a NumPy array representing an image")

	# Apply PCA
	pca = PCA(n_components=n_components, whiten=True)
	X_pca = pca.fit_transform(image)
	X_pca = X_pca.flatten().tolist()

	return X_pca


def dct(image):
	# If input is a string, assume it's a file path
	if isinstance(image, str):
		image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (resize, resize))
	# If input is a NumPy array, assume it's an image
	elif isinstance(image, np.ndarray):
		image = image
	else:
		raise ValueError("Input must be either a file path or a NumPy array representing an image")

	# Apply DCT to the image
	dct_image = cv2.dct(np.float32(image))
	dct_image = dct_image[:16, :16]

	# Flatten the DCT coefficients to create the feature vector
	dct_features = dct_image.flatten().tolist()

	return dct_features


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
			cv2.circle(img_cv, (part.x, part.y), 10, (0, 255, 0), -1)

		cv2.imwrite("face_shape_output.jpg", img_cv)
		cv2.imwrite("face_aligned_output.jpg", face_aligned)

	return face_descript


def yolo_landmark(
	img_path: str,
	xyxy: tuple,
	dlib_recognizer: str,
	dlib_landmark: str,
	save_img: bool = False):
	"""
	return:
	encodings
	names
	"""
	# feature container
	recognizer = dlib.face_recognition_model_v1(dlib_recognizer)
	shape_predictor = dlib.shape_predictor(dlib_landmark)

	# face box
	face = dlib.load_rgb_image(img_path)
	left, top, right, bottom = xyxy
	dlibRect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
	
	# get the shape of the face
	face_shape = shape_predictor(face, dlibRect)
	face_aligned = dlib.get_face_chip(face, face_shape, size=150, padding=0.25)

	# if save_img == True, save the image with the facial landmarks
	# Convert dlib's image object to OpenCV's image format
	if save_img:
		img_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
		for part in face_shape.parts():
			# Draw a circle at each landmark
			cv2.circle(img_cv, (part.x, part.y), 10, (0, 255, 0), -1)

		cv2.imwrite("face_shape_output.jpg", img_cv)
		cv2.imwrite("face_aligned_output.jpg", face_aligned)
	
	# get the facial feature extraction
	face_descript = recognizer.compute_face_descriptor(face_aligned)

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