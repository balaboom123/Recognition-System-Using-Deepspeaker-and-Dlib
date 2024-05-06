"""
Usage:
  face_recognize.py -d <train_dir> -i <test_image>

Options:
  -h, --help                     Show this help
  -d, --train_dir =<train_dir>   Directory with
                                 images for training
  -i, --test_image =<test_image> Test image
"""

# importing libraries
import face_recognition
# import docopt
import dlib
from sklearn import svm
import os
import cv2
import numpy as np
from imutils.face_utils import rect_to_bb


def get_facial_feature(dir):
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
            print(dir + person + "/" + person_img)

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


def svc_classify(encodings, names, tests_dir):
    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)

    if tests_dir[-1] != '/':
        tests_dir += '/'
    tests_img = os.listdir(tests_dir)
    # Load the test image with unknown faces into a numpy array
    for test_img in tests_img:
        test_image = face_recognition.load_image_file(tests_dir + '/' + test_img)

        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(test_image)
        no = len(face_locations)
        print("Number of faces detected: ", no)

        # Predict all the faces in the test image using the trained classifier
        print("Found:")
        for i in range(no):
            test_image_enc = face_recognition.face_encodings(test_image)[i]
            name = clf.predict([test_image_enc])
            print(*name)


# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 bbb 文件夹的路径
train_dir = os.path.join(script_dir, "data/train_data/image")
test_dir = os.path.join(script_dir, "data/test_data")
feature, person = get_facial_feature(train_dir)
print(feature)
print(person)