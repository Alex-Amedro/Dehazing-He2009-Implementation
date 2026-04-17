import numpy as np
import matplotlib.pyplot as plt

from darkchannelcomputation import darkchannelcomputation as dcp

from atmoslight import atmoslight as al

def transmission_estimation(I : np.ndarray, patch_size : int, omega=0.95) -> np.ndarray:

    dark_channel = dcp(I,patch_size)

    A = al(dark_channel,I)

    # Normalisation

    I_copie = I.copy()

    I_normee = np.divide(I_copie,A)

    I_filtree = dcp(I_normee,patch_size)

    
    return np.ones(shape=np.shape(I_filtree)) - omega * I_filtree


if __name__ == "__main__":
    img = plt.imread("../../data/raw_images/image1.jpg")
    transminsion = transmission_estimation(img, 50)
    fig = plt.figure()
    original = fig.add_subplot(1, 2, 1)
    original.imshow(img)
    dark = fig.add_subplot(1, 2, 2)
    dark.imshow(transminsion, cmap='gray')
    plt.show()