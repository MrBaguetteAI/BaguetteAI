import json
import numpy as np
import matplotlib.pyplot as plt


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path, inputs):
    with open(path) as f:
        data = json.load(f)
    return data

def RGB_seggrigation(d):
    R = d[:, :, 0]
    G = d[:, :, 1]
    B = d[:, :, 2]

    write_D(R,"R_data.json")
    write_D(G,"G_data.json")
    write_D(B,"B_data.json")

    return R,G,B

def RGB_conjugation(R,G,B):
    #Combines all the arrays into one array ex: (200, 300, 3)
    return np.stack((R,G,B),axis=-1) 


def read_img(image):
    image_array = plt.imread(image)
    return image_array

def show_img(data,color):
    match color:
        case "R":
            plt.imshow(data,cmap='Reds')
            plt.show()
        
        case "G":
            plt.imshow(data,cmap='Greens')
            plt.show()

        case "B":
            plt.imshow(data,cmap='Blues')
            plt.show()

        case "F":
            plt.imshow(data)
            plt.show()

def filter_apply(D, fil):
    d = np.array(D, dtype=float)
    fil = np.array(fil, dtype=float)
    filt = fil.shape[0]

    H, W = d.shape
    out_h, out_w = H - filt + 1, W - filt + 1

    # Build output array
    activation_M = np.empty((out_h, out_w))

    # Vectorized inner product using broadcasting
    for i in range(out_h):
        for j in range(out_w):
            activation_M[i, j] = np.sum(d[i:i+filt, j:j+filt] * fil)

    return activation_M


def pooling(d, stride=2, filt=2):
    # Ensure dimensions are divisible by filt
    h, w = d.shape
    h_out, w_out = h // filt, w // filt

    # Reshape into blocks of size filt×filt
    d = d[:h_out*filt, :w_out*filt]  # crop if not divisible
    d = d.reshape(h_out, filt, w_out, filt)

    # Take mean over the small blocks
    activation_M = d.mean(axis=(1, 3))

    return activation_M
