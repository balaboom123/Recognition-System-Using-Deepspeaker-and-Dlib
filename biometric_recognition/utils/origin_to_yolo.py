import dlib
from sklearn import svm
import os
import cv2
import numpy as np
import yaml
from imutils.face_utils import rect_to_bb
import random
import re

from utils.root_path import root
import utils.coordinate


def find_cls(my_dict, file_name):
    """
    find the class name from file_name

    :param my_dict:
    :param file_name:
    :return:
    """

    for key, value in my_dict.items():
        if value == file_name:
            return key
    else:
        raise Exception("can't find the class name")


class VideoToYoloFormat:

    def __init__(self, file_name, video_path, img_output):
        # init path
        self.video_path = video_path
        self.img_output = img_output
        self.file_name = file_name
        print("file_name:", self.file_name)
        print("video_path:", self.video_path)
        print("img_output:", self.img_output)


        # init detector
        detector_model = f"{root}/model/mmod_human_face_detector.dat"
        self.detector = dlib.cnn_face_detection_model_v1(detector_model)

        # check output directory
        require_dir = [self.img_output,
                       f"{self.img_output}/train",
                       f"{self.img_output}/val",
                       f"{self.img_output}/train/images",
                       f"{self.img_output}/train/labels",
                       f"{self.img_output}/val/images",
                       f"{self.img_output}/val/labels"]

        for dir_ in require_dir:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        print("sucessfully init VideoToYoloFormat")
        print("==================================")

    def one_face_rect_detect(self, img):
        """
check whether it is one face in the image
if one face in the image, return the rect
else return False

:param img:
:return: rect
"""
        if isinstance(img, np.ndarray):
            img_filename = "temp_image.png"
            dlib.save_image(img, img_filename)

            # load img to dlib
            face = dlib.load_rgb_image(img_filename)

        elif isinstance(img, str):
            face = dlib.load_rgb_image(img)
        else:
            raise Exception("img must be numpy.ndarray or str")

        # detect face
        detection = self.detector(face, 0)

        # reture whether it is one face in the image and the rect
        if len(detection) != 1:
            return False, None
        else:
            rect = detection[0].rect
            return True, rect

    def video_to_face_image(self, time_interval=1, val_period=6):
        """
        save the face image from video

        :param val_period:
        :param time_interval:
            the interval of time(sec) to save
        :return: path_rect
        """
        print("now start processing:", self.file_name)
        # establish path and rect dict
        path_rect = {}

        # read video
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        val_period_count = 1
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print("video_to_face_image start:", self.file_name)
        while cap.isOpened():
            frame_count += 1
            ret, frame = cap.read()

            # if no frame, break
            if not ret:
                break

            # if not the time_interval frame, skip
            if frame_count % (fps*time_interval) != 0:
                continue

            # dir and file path
            train_img_dir = f"{self.img_output}/train/images"
            train_img_file = os.path.join(train_img_dir, f"{self.file_name}_{frame_count}.png")

            train_cls_dir = f"{self.img_output}/train/labels"
            train_cls_file = os.path.join(train_cls_dir, f"{self.file_name}_{frame_count}.txt")

            val_img_dir = f"{self.img_output}/val/images"
            val_img_file = os.path.join(val_img_dir, f"{self.file_name}_{frame_count}.png")

            val_cls_dir = f"{self.img_output}/val/labels"
            val_cls_file = os.path.join(val_cls_dir, f"{self.file_name}_{frame_count}.txt")

            # check whether the file exists (improve the speed)
            train_exist = os.path.exists(train_img_file) and os.path.exists(train_cls_file)
            val_exist = os.path.exists(val_img_file) and os.path.exists(val_cls_file)
            if train_exist or val_exist:
                continue

            # check whether it is one face in the image and get the face rect
            is_one_face, face_rect = self.one_face_rect_detect(frame)

            # if one face in the frame, save the frame
            if is_one_face and val_period_count % val_period != 0:
                # count and save to train data
                val_period_count += 1

                cv2.imwrite(train_img_file, frame)
                path_rect[train_img_file] = face_rect

            elif is_one_face and val_period_count % val_period == 0:
                # count and save to val data
                val_period_count = 1

                cv2.imwrite(val_img_file, frame)
                path_rect[val_img_file] = face_rect

        print("video_to_face_image done:", self.file_name)
        cap.release()

        return path_rect

    def yolo_label(
            self,
            img_path: str,
            cls: int,
            pos: tuple):
        """
        generate yolov8 label .txt

        :param cls: class name
        :param pos: (x, y, width, height): must be normalized
        :return:
        """
        x, y, width, height = pos

        content = f"{cls} {x} {y} {width} {height}"

        file_path, _ = os.path.splitext(img_path)
        label_path = file_path.replace("images", "labels") + ".txt"
        if not os.path.exists(label_path):
            # write in .txt
            with open(label_path, "w") as f:
                f.write(content)

            print("label done:", label_path)
