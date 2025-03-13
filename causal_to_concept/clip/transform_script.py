import torch
import clip
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import h5py
import sys
import torchvision
import os


def embed_data(n, dir, name):
    total_splits = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(name, device=device)

    dataset = h5py.File('3dshapes.h5', 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}

    lower = (n_samples // total_splits) * n
    upper = (n_samples // total_splits) * (n + 1)
    images = images[lower:upper]
    outputs = []
    model.eval()
    for i, image in enumerate(images):
        print("Did {} of {} images".format(i, len(images)))
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0)
        outputs.append(model.encode_image(image).detach().numpy())

    outputs = np.vstack(outputs)
    np.save(os.path.join(dir, 'embeddings_{}.npy'.format(n)), outputs)


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <n>")
        sys.exit(1)

    # Parse the argument
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: n must be an integer")
        sys.exit(1)

    # Call embed_data with the parsed argument

    dir = './data_transformed'

    name = "ViT-B/32"
    name = 'RN101'
    if not os.path.exists(os.path.join(dir, 'embeddings_{}.npy'.format(n))):
        embed_data(n, dir, name)
    else:
        print("already done")


if __name__ == "__main__":
    main()