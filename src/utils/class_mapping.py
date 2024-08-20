import json
import os


def load_class_mapping(class_mapping_path):
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at {class_mapping_path}")

    with open(class_mapping_path, 'r') as f:
        class_to_index = json.load(f)
    return class_to_index
