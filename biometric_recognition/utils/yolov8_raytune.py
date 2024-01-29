from ultralytics import YOLO
from utils.root_path import root
import os


def train_yolo():
	# Define a YOLO model
	model = YOLO("yolov8n.pt")

	# Run Ray Tune on the model
	yaml_path = f"{root}/data/yolo_data/data.yaml"
	result_grid = model.tune(
		data=yaml_path,
		epochs=30,
		iterations=300,
		optimizer='AdamW')
		#  plots=False, save=False, val=False)


if __name__ == '__main__':
	train_yolo()