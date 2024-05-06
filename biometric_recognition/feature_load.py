import os
import json

def feature_load(input_path: str):

	with open(input_path, "r") as json_file:
		data_dict = json.load(json_file)
	
	return data_dict