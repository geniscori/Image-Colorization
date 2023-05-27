import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import numpy as np


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    plt.clf()  # clear matplotlib

    # Passem a numpy
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()

    # Transposem i reescalem
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100

    # Normalitzem la imatge
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    # Passem a l'escala RGB i tornem a passar-lo a numpy
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()

    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))