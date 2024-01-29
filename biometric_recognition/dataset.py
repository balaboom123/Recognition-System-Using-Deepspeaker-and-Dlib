from utils.origin_to_yolo import *
from utils.root_path import root
from utils.coordinate import *
import os
import yaml
import re

# init path
video_root = f"{root}/data/origin_data"
voice_root = f"{root}/data/voice_data"
img_root = f"{root}/data/yolo_data"

# get the class dict and number of class
names = [os.path.splitext(name)[0].split('_')[1] for name in os.listdir(video_root)]
names = list(set(names))

if os.path.exists(f"{img_root}/data.yaml"):
	with open(f"{img_root}/data.yaml", "r") as f:
		yaml_data = yaml.load(f, Loader=yaml.FullLoader)
		yaml_names = yaml_data['names'].values()
		yaml_cls_num = len(yaml_names)

	for name in names:
		if name not in yaml_names:
			yaml_cls_num += 1
			yaml_data['names'][yaml_cls_num - 1] = name
			yama_data['nc'] = yaml_cls_num

	with open(f"{img_root}/data.yaml", "w") as f:
		yaml.dump(yaml_data, f)

if not os.path.exists(f"{img_root}/data.yaml"):
	class_dict = dict(enumerate(names))
	nc_num = len(names)

	data = dict(
		train=f"./train/images",
		val=f"./val/images",
		nc=nc_num,
		names=class_dict
	)

	with open(f"{img_root}/data.yaml", "w") as f:
		yaml.dump(data, f)


# check output directory
for video in os.listdir(video_root):
	# get video path and file name
	video_path = f"{video_root}/{video}"
	student_id = os.path.splitext(video)[0].split("_")[1]

	# initialize VideoToYoloFormat
	vdyf = VideoToYoloFormat(student_id, video_path, img_root)
	face_rect = vdyf.video_to_face_image()

	for img_path, rect in face_rect.items():
		print("img_path is: ", img_path)
		# get img xywh
		x, y, w, h = rect_to_xywh(rect)

		# get img shape
		img_height, img_width = img_shape(img_path)

		# get normalized xywh
		x_center, y_center, width, height = normalized_xywh((x, y, w, h), img_height, img_width)
		pos = (x_center, y_center, width, height)

		# get label
		with open(f"{img_root}/data.yaml", "r") as f:
			yaml_data = yaml.load(f, Loader=yaml.FullLoader)
			class_dict = yaml_data['names']

		img_name = os.path.basename(img_path)
		img_cls = img_name.split("_")[0]
		cls_name = find_cls(class_dict, img_cls)
		vdyf.yolo_label(img_path, cls=cls_name, pos=pos)

	print("finish :", video)
	print("==================================")



print("Dataset setup")

