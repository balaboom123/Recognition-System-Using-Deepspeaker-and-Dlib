import os
import json
import dlib

from biometric_recognition.face_feature_extract import landmark
from biometric_recognition.face_feature_extract import face_detect_resize
from biometric_recognition.feature_process import retain_avg_vector
from biometric_recognition.feature_save import feature_save, check_reapetition

# face feature extraction
# root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
person_root = f"{script_dir}/data/person_data"

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

	vectors = feature_dict["landmark_feature_68"]
	vectors = retain_avg_vector(vectors)
	feature_dict["landmark_feature_68"] = vectors

	if len(feature_dict["file_name"]) > 0:
		feature_save(person_json, feature_dict)


