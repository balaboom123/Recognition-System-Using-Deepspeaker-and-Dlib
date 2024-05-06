from biometric_recognition.utils.video_dataset import video_to_images
import os
import cv2

# root
script_dir = os.path.dirname(os.path.abspath(__file__))
file_root = f"{script_dir}/data/origin_data"
output_root = f"{script_dir}/data/person_data"

file_names = os.listdir(file_root)
for file_name in file_names:
	file_path = os.path.join(file_root, file_name)

	person_name = file_name.replace("Face_", "").replace(".mp4", "")
	person_name_path = os.path.join(output_root, person_name)
	if not os.path.exists(person_name_path):
		os.mkdir(person_name_path)
		video_to_images(file_path, person_name_path, 0.25)