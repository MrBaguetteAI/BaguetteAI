import json
import csv

def read_csv(path):
    with open(path, mode='r', newline='', encoding='utf-8') as file:
        data = csv.reader(file)
        next(data, None)  # Skips the header row safely
        yield from data

def write_json(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=8)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

