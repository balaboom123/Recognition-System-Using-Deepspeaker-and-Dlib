from ultralytics import YOLO
from utils.root_path import root
from sklearn.model_selection import ParameterGrid
import os
import json

def train_evaluate_yolo(param):
	model = YOLO()
	yaml_path = f"{root}/data/yolo_data/data.yaml"
	model.train(data=yaml_path, save=False, val=False, **param)  # train the model
	metrics = model.val()

	box_mAP = metrics.box.map  # map50-95
	box_mAP50 = metrics.box.map50  # map50
	box_mAP75 = metrics.box.map75  # map75
	box_mAPs = metrics.box.maps

	return box_mAP, box_mAP50, box_mAP75, box_mAPs


if __name__ == '__main__':
	# Define hyperparameters to tune
	learning_rates = [0.5, 0.1]  # 0.005,
	batch_sizes = [8]  # , 16, 32
	optimizers = ['auto']
	epochs = [25]  # , 50, 100
	network_architectures = ['yolov8s']
	# activation_functions = ['relu', 'leaky_relu']
	# loss_functions = ['binary_crossentropy', 'focal_loss']
	data_augmentation = [True]
	"""
	learning_rates = [0.001, 0.01, 0.1]
	batch_sizes = [8, 16, 32, 64, 128]
	optimizers = ['auto']
	epochs = [25, 50, 100, 200]
	network_architectures = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
	# activation_functions = ['relu', 'leaky_relu']
	# loss_functions = ['binary_crossentropy', 'focal_loss']
	data_augmentation = [True]
	"""

	# Set up the search space
	search_space = {
		'learning_rate': learning_rates,
		'batch_size': batch_sizes,
		'optimizer': optimizers,
		'epochs': epochs,
		'network_architecture': network_architectures,
		# 'activation_function': activation_functions,
		# 'loss_function': loss_functions,
		'data_augmentation': data_augmentation
	}

	# Create parameter grid
	parameter_grid = ParameterGrid(search_space)

	# Train and evaluate the model for each combination of hyperparameters
	for parameters in parameter_grid:
		# Set hyperparameters for the model
		learning_rate = parameters['learning_rate']
		batch_size = parameters['batch_size']
		optimizer = parameters['optimizer']
		num_epochs = parameters['epochs']
		architecture = parameters['network_architecture']
		# activation = parameters['activation_function']
		# loss = parameters['loss_function']
		augmentation = parameters['data_augmentation']

		parameters = {
			'lr0': learning_rate,
			'batch': batch_size,
			'optimizer': optimizer,
			'epochs': num_epochs,
			'model': architecture,
			# 'activation': activation,
			# 'loss': loss,
			'augment': augmentation}

		if os.path.exists('maps_params.json'):
			with open('maps_params.json', 'r') as file:
				load = json.load(file)
				maps_params = load["maps_params"]

			for map_param in maps_params:
				if map_param["params"] == parameters:
					continue

		# Train and evaluate the model
		mAP, mAP50, mAP75, mAPS = train_evaluate_yolo(parameters)

		save = {
			"maps": {
				'mAP': mAP,
				'mAP50': mAP50,
				'mAP75': mAP75,
				'mAPS': mAPS
			},

			"params": {
				'lr0': learning_rate,
				'batch': batch_size,
				'optimizer': optimizer,
				'epochs': num_epochs,
				'model': architecture,
				# 'activation': activation,
				# 'loss': loss,
				'augment': augmentation
			}
		}

		if not os.path.exists('maps_params.json'):
			with open('maps_params.json', 'w') as file:
				save = {"maps_params": []}
				json.dump(save, file, indent=4)

		if os.path.exists('maps_params.json'):
			with open('maps_params.json', 'r') as file:
				load = json.load(file)
				map_params = load["maps_params"]

			with open('maps_params.json', 'w') as file:
				save = map_params.append(save)
				json.dump(save, file, indent=4)


def valid_yolo():
	# Load a model
	model = YOLO(r"C:\Users\user\Github\biometric_recognition\biometric_recognition\runs\detect\train1\weights\best.pt")

	# Validate the model
	metrics = model.val()  # no arguments needed, dataset and settings remembered
	print(metrics.box.map)  # map50-95
	print(metrics.box.map50)  # map50
	print(metrics.box.map75)  # map75
	print(metrics.box.maps)  # a list contains map50-95 of each category
