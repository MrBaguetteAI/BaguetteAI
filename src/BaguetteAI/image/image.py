import numpy as np
from PIL import Image

def read_img(image):
    img = Image.open(image)
    channel = img.mode
    match channel:
        case "RGB":
            data = dict(width = img.size[0],
                        height = img.size[1],
                        r=list(img.getchannel("R").tobytes()),
                        g=list(img.getchannel("G").tobytes()),
                        b=list(img.getchannel("B").tobytes()),
                        )          # an "L"-mode image of just alpha
                                   # bytes object, each 0-255, index = i
        case "RGBA":
            data = dict(width = img.size[0],
                        height = img.size[1],
                        r=list(img.getchannel("R").tobytes()),
                        g=list(img.getchannel("G").tobytes()),
                        b=list(img.getchannel("B").tobytes()),
                        a=list(img.getchannel("A").tobytes())
                        )          # an "L"-mode image of just alpha
                                   # bytes object, each 0-255, index = i
    return data

def show_img(data,color):
    match color:
        case "r" | "R":
            # Isolate Red: (R, 0, 0, 255)
            rgba_tuples = [(r, 0, 0, 255) for r in data["r"]]
            
            img = Image.new("RGBA", (data["width"], data["height"]))
            img.putdata(rgba_tuples)
            img.show()       
    
        case "g" | "G":
            # Isolate Green: (0, G, 0, 255)
            rgba_tuples = [(0, g, 0, 255) for g in data["g"]]
            
            img = Image.new("RGBA", (data["width"], data["height"]))
            img.putdata(rgba_tuples)
            img.show()
    
        case "b" | "B":
            # Isolate Blue: (0, 0, B, 255)
            rgba_tuples = [(0, 0, b, 255) for b in data["b"]]
            
            img = Image.new("RGBA", (data["width"], data["height"]))
            img.putdata(rgba_tuples)
            img.show()
    
        case "a" | "A":
            # Isolate Alpha: (0, 0, 0, A)
            rgba_tuples = [(a, a, a, 0) for a in data["a"]]
            
            img = Image.new("RGBA", (data["width"], data["height"]))
            img.putdata(rgba_tuples)
            img.show()

        case "f" | "F":
            # Full Color: Zip R, G, and B lists together with full opacity
            rgba_tuples = list(zip(data["r"], data["g"], data["b"], data["a"]))
            
            img = Image.new("RGBA", (data["width"], data["height"]))
            img.putdata(rgba_tuples)
            # img.save("photo_modified.png")
            img.show()

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
