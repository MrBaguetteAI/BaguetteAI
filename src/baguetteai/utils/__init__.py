from .utils import (
        load_json,
        save_json,
        RGB_seggrigation,
        RGB_conjugation,
        read_img,
        show_img,
        filter_apply,
        pooling,
        )

__all__ = [
        "load_json", "save_json","RGB_seggrigation", "RGB_conjugation","read_img","show_img", "filter_apply", "pooling",
        ]


'''
load_json -> to read data from a json file
save_json -> to write the inputed data in the json file
RGB_seggrigation -> to distribute the R, G, and B values in respective lists
RGB_conjugation -> to combine all the R, G, and B lists into one lists
read_img -> used to read an image and get the list of the values
show_img -> to plot the image list using matplotlib
filter_apply -> used to apply the filter on a 2D list of the image
pooling -> used for the pooling step of the CNN
'''
