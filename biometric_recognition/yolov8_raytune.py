from ultralytics import YOLO
import os


def train_yolo():
	root = os.path.dirname(os.path.abspath(__file__))
	model = YOLO("yolov8s.yaml")
	yaml_path = f"{root}/data/yolo_faces/data.yaml"
	"""
	model.train(
		data=yaml_path,
		epochs=100,
		batch=16,
		lr0=0.008,
		lrf=0.008
	)  # train the model
	"""


	model.tune(
		data=yaml_path, 
		epochs=30, 
		iterations=300, 
		optimizer='AdamW', 
		plots=False, 
		save=False, 
		val=False)



if __name__ == '__main__':
	train_yolo()