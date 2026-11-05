from ._core import fast_process  # Import the compiled C++ function
from .util import (
        load_json,
        save_json,
        )

# You can create a pure Python wrapper if you want to keep the API identical
def process_data(data, factor=2.0):
    return fast_process(data, factor)

__all__ = [
        "load_json", "save_json", 'process_data',
        ]


'''
load_json -> to read data from a json file
save_json -> to write the inputed data in the json file
'''
