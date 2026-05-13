import numpy as np
import math
import matplotlib.pyplot as plt
import os
import darkchannelcomputation
from scipy import ndimage, datasets
from cache_utils import load_cache, save_cache, get_img_name


def atmoslight(dark_channel : np.ndarray, I : np.ndarray, img_name : str = None) -> np.ndarray:

    # Si un nom d'image est fourni, vérifier le cache
    if img_name is not None:
        cached = load_cache(img_name, 'atmoslight')
        if cached is not None:
            return cached

    I_copie = I.copy()

    intensites = dark_channel.flatten() # vectorisation de l'image par lignes ( à vérifier )

    indices_tries = np.argsort(intensites)

    indices_tries = np.flip(indices_tries) # par ordre décroissant d'intensitées

    nb_valides = 0.001 * np.shape(indices_tries)[0]

    nb_valides = round(nb_valides)

    #print(nb_valides)

    valides = indices_tries[:nb_valides]

    coordonnees = np.zeros((nb_valides,2),dtype=int)

    for i, value in enumerate(valides):
        y = value % np.shape(I)[1]
        x = value // np.shape(I)[1]
        coordonnees[i] = [x,y]

    # a finir

    #print(np.shape(coordonnees))

    pixels_candidats = I[coordonnees[:,0], coordonnees[:,1]]

    valeurs = np.sum(pixels_candidats, axis=1)

    idx_max = np.argmax(valeurs)

    #print(coordonnees[idx_max,0], coordonnees[idx_max,1])

    # fig = plt.figure()
    # original = fig.add_subplot(1, 2, 1)
    # original.imshow(I)
    # original.set_title("Image originale")
    
    # dark = fig.add_subplot(1, 2, 2)
    # dark.imshow(I)
    # dark.plot(coordonnees[idx_max,1], coordonnees[idx_max,0], 'ro', markersize=20, fillstyle='none', markeredgewidth=2)
    # dark.set_title("Lumière atmosphérique")
    # plt.show()

    result = pixels_candidats[idx_max]

    # Sauvegarder dans le cache
    if img_name is not None:
        save_cache(img_name, 'atmoslight', result)

    return result

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/raw_images/image1.jpg")
    img_name = get_img_name(img_path)
    img = plt.imread(img_path)
    atmoslight(darkchannelcomputation.darkchannelcomputation(img, 50, img_name), img, img_name)
