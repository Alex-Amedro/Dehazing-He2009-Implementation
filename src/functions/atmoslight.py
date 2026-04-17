import numpy as np
import math
import matplotlib.pyplot as plt
import darkchannelcomputation
from scipy import ndimage, datasets


def atmoslight(dark_channel : np.ndarray, I : np.ndarray) -> np.ndarray:

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

    return pixels_candidats[idx_max]

if __name__ == "__main__":
    img = plt.imread("data/raw_images/image1.jpg")
    atmoslight(darkchannelcomputation.darkchannelcomputation(img, 50), img)
