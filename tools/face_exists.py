def get_facial_feature(dir):
	# root dir
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# load model
	detector_model = os.path.join(script_dir, "model\\mmod_human_face_detector.dat")
	recognizer_model = os.path.join(script_dir, "model\\dlib_face_recognition_resnet_model_v1.dat")
	shape_predict_model = os.path.join(script_dir, "model\\shape_predictor_68_face_landmarks.dat")

	detector = dlib.cnn_face_detection_model_v1(detector_model)
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

			# detect face
			detection = detector(face, 0)

			if detection[0]:
				# get the shape of the face
				face_shape = shape_predictor(face, detection[0].rect)
				face_aligned = dlib.get_face_chip(face, face_shape, size=150, padding=0.25)

				# get the facial feature extraction
				face_descript = recognizer.compute_face_descriptor(face_aligned)
				face_descript = np.array(face_descript)
				encodings.append(face_descript)
				names.append(person)

	return encodings, names