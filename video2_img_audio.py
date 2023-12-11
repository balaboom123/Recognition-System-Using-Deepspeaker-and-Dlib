import cv2
import os
from moviepy.editor import VideoFileClip
from pathlib import Path


def video_to_frames(video_file, output_dir, file_name, frame_interval=1):
	# 检查输出目录是否存在，如果不存在则创建
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# 读取视频文件
	cap = cv2.VideoCapture(video_file)
	count = 0

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# 保存每帧图片
		file_dir = os.path.join(output_dir, f"image_{file_name}_{count:05d}.png")
		if not os.path.exists(file_dir) and count % frame_interval == 0:
			cv2.imwrite(file_dir, frame)
		count += 1

	cap.release()
	print(f"Total frames extracted: {count}")


def extract_audio(video_file, output_dir, file_name):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	video = VideoFileClip(video_file)
	audio = video.audio

	file_dir = os.path.join(os.path.join(output_dir, f"voice_{file_name}.wav"))
	if not os.path.exists(file_dir):
		audio.write_audiofile(file_dir)
		audio.close()


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "train_data/")
print(data_dir)
# 使用示例
for video in os.listdir(f"{data_dir}/Face"):
	print(video)
	student_id = video.split(".")[0].split("_")[1]
	video_to_frames(f"{data_dir}/Face/{video}", f"{data_dir}/image/{student_id}", student_id, 20)
	extract_audio(f"{data_dir}/Face/{video}", f"{data_dir}/voice", student_id)
