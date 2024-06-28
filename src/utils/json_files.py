import os
import json


def load_json_file(filename):
    f = open(filename)
    loaded_json_file = json.load(f)
    f.close()

    return loaded_json_file


def check_dir_and_save_json(output_dir, filename, output_data):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_dir = output_dir + '\\' + filename
    with open(filename_dir, 'w') as f:
        json.dump(output_data, f)
