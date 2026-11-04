import json

def save_json(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=8)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

