import cv2
import dlib
import numpy as np
import os
# import face_recognition


def delete_mutiface_picture(dir):
	# Training the SVC classifier
	# The training data would be all the
	# face encodings from all the known
	# images and the labels are their names
	encodings = []
	names = []
	script_dir = os.path.dirname(os.path.abspath(__file__))
	mmod_human_face_detector = os.path.join(script_dir, "mmod_human_face_detector.dat")

	# Training directory
	if dir[-1] != '/':
		dir += '/'
	train_dir = os.listdir(dir)

	# Loop through each person in the training directory
	for person in train_dir:
		pix = os.listdir(dir + person)

		# Loop through each training image for the current person
		for person_img in pix:
			# Get the face encodings for the face in each image file
			# face = face_recognition.load_image_file(dir + person + "/" + person_img)
			# face_bounding_boxes = face_recognition.face_locations(face)

			face = dlib.load_rgb_image(dir + person + "/" + person_img)
			# 將圖像轉換為灰度圖像
			gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			# 50% -> 44 s/per，40% -> 30 s/per
			scale_percent = 40
			width = int(gray_face.shape[1] * scale_percent / 100)
			height = int(gray_face.shape[0] * scale_percent / 100)
			dim = (width, height)
			resized_gray_face = cv2.resize(gray_face, dim)

			dnnFaceDetector = dlib.cnn_face_detection_model_v1(mmod_human_face_detector)
			rects = dnnFaceDetector(resized_gray_face, 0)
			print(f'sucess process{rects}')

			# If training image contains not exactly one face, delete it
			if len(rects) != 1:
				print(f"Deleting {person}/{person_img} as it has {len(rects)} faces.")
				# os.remove(dir + person + "/" + person_img)
			else:
				print(f"Retain {person}/{person_img} as it has {len(rects)} faces.")
	return "success"






# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 bbb 文件夹的路径
data_dir = os.path.join(script_dir, "data")

# Example usage
delete_mutiface_picture(data_dir)
