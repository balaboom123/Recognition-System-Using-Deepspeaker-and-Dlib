import cv2
import os
from moviepy.editor import VideoFileClip
from pathlib import Path
from utils.root_path import root
from utils.get_one_face import get_one_face



def extract_audio(video_file, output_dir, file_name):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	video = VideoFileClip(video_file)
	audio = video.audio

	file_dir = os.path.join(os.path.join(output_dir, f"voice_{file_name}.wav"))
	if not os.path.exists(file_dir):
		audio.write_audiofile(file_dir)
		audio.close()


data_dir = os.path.join(root, "train_data/")

# 使用示例
for video in os.listdir(f"{data_dir}/Face"):
	print(video)
	student_id = video.split(".")[0].split("_")[1]
	video_to_frames(f"{data_dir}/Face/{video}", f"{data_dir}/image/{student_id}", student_id, 20)
	extract_audio(f"{data_dir}/Face/{video}", f"{data_dir}/voice", student_id)
