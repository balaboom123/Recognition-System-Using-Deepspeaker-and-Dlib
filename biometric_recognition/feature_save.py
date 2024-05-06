import os
import json


def init_dict(output_path):
    try:
        with open(output_path, "r") as json_file:
            data_dict = json.load(json_file)
    except:
        # Handle JSONDecodeError (e.g., file is empty or not in correct format)
        data_dict = {}
    return data_dict


def dict_write_value(read_data, write_dict):
    # Append new data to existing contents
    for key, value in write_dict.items():
        if key in read_data:
            read_data[key].append(value)
        else:
            read_data[key] = value

    return read_data


def check_reapetition(json_dict, name: dict):
    if name in json_dict["file_name"]:
        return False
    else:
        return True


def feature_save(output_path: str, data_dict: dict):
    # Read existing JSON file if it exists
    existing_data_dict = init_dict(output_path)

    write_dict = dict_write_value(existing_data_dict, data_dict)

    with open(output_path, "w") as json_file:
        json.dump(write_dict, json_file, indent=4)