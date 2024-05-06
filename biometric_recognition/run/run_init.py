import dlib
import os
import json
import numpy as np
from deep_speaker.conv_models import DeepSpeakerModel
import cv2
import pyaudio
import soundfile as sf
import random

def init_model(rec_path, sp_path, ds_path):
	# dlib
	recognizer = dlib.face_recognition_model_v1(rec_path)
	shape_predictor_68 = dlib.shape_predictor(sp_path)
	detector = dlib.get_frontal_face_detector()

	# deep speaker
	deep_speaker = DeepSpeakerModel()
	deep_speaker.m.load_weights(ds_path, by_name=True)

	return recognizer, shape_predictor_68, detector, deep_speaker


def init_recorder(cam_num=0):
	# audio
	audio_record = pyaudio.PyAudio()

	# Initialize the video capture
	# Use 0 for the default camera, or specify a different index for other cameras
	cap = cv2.VideoCapture(cam_num)

	return audio_record, cap


def init_features(person_root):
	# def dict
	x = []
	y = []

	all_entries = os.listdir(person_root)
	person_names = [
		entry for entry in all_entries if os.path.isdir(
			os.path.join(
				person_root,
				entry))]

	for person_name in person_names[:3]:
		vectors = []
		# person_name_path = os.path.join(person_root, person_name)
		person_json = f"{person_root}/{person_name}.json"

		with open(person_json, "r") as json_file:
			data_dict = json.load(json_file)

		y.extend([person_name])
		length = len(data_dict["landmark_feature_68"])
		for i in range(length):
			vectors.extend([data_dict["landmark_feature_68"][i]])
		avg_vectors = np.mean(vectors, axis=0)
		x.extend([avg_vectors])

	return x, y


def init_question(dir):
	speech_list = [os.path.splitext(f) for f in os.listdir(dir) if f.endswith(".wav")]
	speech_list = [name for name, _ in speech_list]
	random_element = random.choice(speech_list)

	return random_element