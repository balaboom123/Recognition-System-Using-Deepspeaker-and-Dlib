from biometric_recognition.face_feature_extract import landmark, face_detect_resize, face_encoding
from biometric_recognition.feature_save import feature_save, check_reapetition
from biometric_recognition.voice_feature_extract import mfcc, lpc, lpcc
from biometric_recognition.face_feature_extract import pca, dct, landmark
from biometric_recognition.face_feature_extract import face_detect_resize, face_encoding
from biometric_recognition.feature_process import retain_avg_vector
from biometric_recognition.feature_save import feature_save, check_reapetition

from PIL import Image
import numpy as np
import os
import cv2
import dlib
from ultralytics import YOLO
import cv2
import json
from moviepy.editor import VideoFileClip
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def extract_and_boost_audio(video_path, output_audio, boost_factor):
    # 加載影片
    clip = VideoFileClip(video_path)

    # 獲取影片的音頻部分
    audio = clip.audio

    # 提升音量
    boosted_audio = audio.volumex(boost_factor)

    # 保存提升音量後的音頻為WAV格式
    boosted_audio.write_audiofile(output_audio, codec='pcm_s16le')

from rich.console import Console
from rich.panel import Panel

# 创建控制台对象
console = Console()

text = ("Hello, this is some text!"
        "This is a new line of text!")

# 创建面板对象，设置文本内容和标题
panel = Panel(text, title="My Panel")

# 打印面板
console.print(panel)

text = ("Hello, this is some text!"
        "This is a new line of text!")
panel = Panel(text, title="My Panel")
console.print(panel)
"""
scripts_dir = os.path.dirname(os.path.abspath(__file__))
input_root = os.path.join(scripts_dir, "data/test")
output_root = os.path.join(scripts_dir, "data/test")
if not os.path.exists(output_root):
    os.mkdir(output_root)

for file in os.listdir(input_root):
    input_path = os.path.join(input_root, file)

    file_name = os.path.basename(input_path)
    path_name, _ = os.path.splitext(file_name)

    # 袋檢查為何去掉.split("_")[1]才能儲存
    output_name = path_name.split("_")[1]

    output_dir = os.path.join(output_root, output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, output_name + ".wav")

    # video_to_images(input_path, output_dir)
    extract_and_boost_audio(input_path, output_path, boost_factor=2)
"""
# root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
# person_root = f"{script_dir}/data/temp"
person_root = f"{script_dir}/data/person_data"


"""
# def dict
x = []
y = []

all_entries = os.listdir(person_root)
person_names = [
    entry for entry in all_entries if os.path.isdir(
        os.path.join(
            person_root,
            entry))]

for person_name in person_names[:11]:
    # person_name_path = os.path.join(person_root, person_name)
    person_json = f"{person_root}/{person_name}.json"

    with open(person_json, "r") as json_file:
        data_dict = json.load(json_file)

    length = len(data_dict["landmark_feature_68"])
    for i in range(length):
        y.extend([person_name])
        x.extend([data_dict["landmark_feature_68"][i]])

# 初始化PCA，設置目標維度為2
pca = PCA(n_components=2)

# 對特徵向量進行降維
features_2d = pca.fit_transform(x)

# 將每個類別的特徵向量分開
class_dict = {}
for i, label in enumerate(y):
    if label not in class_dict:
        class_dict[label] = []
    class_dict[label].append(features_2d[i])

# 繪製散點圖
plt.figure(figsize=(15, 12))  # 調整圖片大小
for label, data in class_dict.items():
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], label=label)

plt.title('PCA Visualization', fontsize=16)  # 調整標題字體大小
plt.xlabel('Principal Component 1', fontsize=14)  # 調整 x 軸標籤字體大小
plt.ylabel('Principal Component 2', fontsize=14)  # 調整 y 軸標籤字體大小
plt.legend(loc='best', fontsize=12)  # 調整圖例位置和字體大小
plt.show()
"""


"""
x = []
y = []
temp_x = []

all_entries = os.listdir(person_root)
person_names = [
    entry for entry in all_entries if os.path.isdir(
        os.path.join(
            person_root,
            entry))]

for person_name in person_names:
    temp = []
    person_name_path = os.path.join(person_root, person_name)

    person_json = f"{person_root}/{person_name}.json"

    with open(person_json, "r") as json_file:
        data_dict = json.load(json_file)

    length = len(data_dict["landmark_feature_68"])
    y.extend([person_name])
    for i in range(length):
        temp.extend([data_dict["landmark_feature_68"][i]])

    # 计算平均向量
    sum_vector = np.sum(temp, axis=0)
    average_vector = sum_vector / len(temp)
    x.extend([average_vector])


for i in range(len(x)):
    print(y[i])
    for j in range(len(x)):
        vector1 = np.array(x[i])
        vector2 = np.array(x[j])

        # 计算两个向量的欧氏距离
        euclidean_distance = np.linalg.norm(vector2 - vector1)
        print(y[j], euclidean_distance)
    print()
"""
