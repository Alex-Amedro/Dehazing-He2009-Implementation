import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from scipy import ndimage, datasets
from cache_utils import load_cache, save_cache, get_img_name


def darkchannelcomputation(I : np.ndarray, patch_size : int, img_name : str = None) -> np.ndarray:

    # Si un nom d'image est fourni, vérifier le cache
    if img_name is not None:
        cached = load_cache(img_name, 'dark_channel')
        if cached is not None:
            return cached

    I_min_canaux = np.min(I, axis=2)

    # I_padded = np.pad(I_min_canaux,patch_size%2, 'constant', constant_values=255)

    dark_channel = ndimage.minimum_filter(I_min_canaux,patch_size)
    

    
    # fig = plt.figure()
    # original = fig.add_subplot(1, 2, 1)
    # original.imshow(I)
    # dark = fig.add_subplot(1, 2, 2)
    # dark.imshow(dark_channel, cmap='gray')
    # plt.show()

    # Sauvegarder dans le cache
    if img_name is not None:
        save_cache(img_name, 'dark_channel', dark_channel)

    return dark_channel



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcul du dark channel d'une image.")
    parser.add_argument("image", type=str, help="Chemin vers l'image d'entrée")
    parser.add_argument("--patch-size", type=int, default=20, help="Taille du patch pour le dark channel (défaut: 20)")
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    img_name = get_img_name(img_path)
    img = plt.imread(img_path)
    dark = darkchannelcomputation(img, args.patch_size, img_name)
    fig = plt.figure()
    original = fig.add_subplot(1, 2, 1)
    original.imshow(img)
    dark_plot = fig.add_subplot(1, 2, 2)
    dark_plot.imshow(dark, cmap='gray')
    plt.show()