import numpy as np

import math

def atmoslight(dark_channel : np.ndarray, I : np.ndarray) -> np.ndarray:

    intensites = dark_channel.flatten() # vectorisation de l'image par lignes ( à vérifier )

    indices_tries = np.argsort()

    indices_tries = np.flip(indices_tries) # par ordre décroissant d'intensitées

    nb_valides = 0.001 * np.shape(indices_tries)[0]

    nb_valides = round(nb_valides)

    valides = indices_tries[:nb_valides]

    coordonnees = np.zeros((nb_valides,2))

    for i, value in enumerate(valides):
        y = value % np.shape(I)[0]
        x = value - y * np.shape(I)[0]
        coordonnees[i] = [x,y]

    # a finir
