import numpy as np
import matplotlib.pyplot as plt
import os

from darkchannelcomputation import darkchannelcomputation as dcp

from atmoslight import atmoslight as al
from cache_utils import load_cache, save_cache, get_img_name

def transmission_estimation(I : np.ndarray, patch_size : int, omega=0.95, img_name : str = None) -> np.ndarray:

    # Si un nom d'image est fourni, vérifier le cache
    if img_name is not None:
        cached_t = load_cache(img_name, 'transmission_estimation')
        cached_A = load_cache(img_name, 'atmoslight')
        if cached_t is not None and cached_A is not None:
            return cached_t, cached_A

    dark_channel = dcp(I, patch_size, img_name)

    A = al(dark_channel, I, img_name)

    # Normalisation

    I_copie = I.copy()

    I_normee = np.divide(I_copie,A)

    I_filtree = dcp(I_normee,patch_size)

    t = np.ones(shape=np.shape(I_filtree)) - omega * I_filtree

    # Sauvegarder dans le cache
    if img_name is not None:
        save_cache(img_name, 'transmission_estimation', t)

    return t, A 


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/raw_images/image1.jpg")
    img_name = get_img_name(img_path)
    img = plt.imread(img_path)
    transminsion, A = transmission_estimation(img, 50, img_name=img_name)
    fig = plt.figure()
    original = fig.add_subplot(1, 2, 1)
    original.imshow(img)
    dark = fig.add_subplot(1, 2, 2)
    dark.imshow(transminsion, cmap='gray')
    plt.show()