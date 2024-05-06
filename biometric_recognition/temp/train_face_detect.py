from ultralytics import YOLO
import os


def train_yolo():
	root = os.path.dirname(os.path.abspath(__file__))
	model = YOLO("yolov8n.yaml")
	yaml_path = f"{root}/data/yolo_faces/data.yaml"
	model.train(
		data=yaml_path,
		epochs=120,
		batch=16,
	)  # train the model


if __name__ == '__main__':
	train_yolo()
