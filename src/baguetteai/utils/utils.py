import json


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path, inputs):
    with open(path) as f:
        data = json.load(f)
    return data
