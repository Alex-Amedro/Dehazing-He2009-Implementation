import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, datasets


def darkchannelcomputation(I : np.ndarray, patch_size : int) -> np.ndarray:

    I_min_canaux = np.min(I, axis=2)

    # I_padded = np.pad(I_min_canaux,patch_size%2, 'constant', constant_values=255)

    dark_channel = ndimage.minimum_filter(I_min_canaux,patch_size)

    # fig = plt.figure()
    # original = fig.add_subplot(1, 2, 1)
    # original.imshow(I)
    # dark = fig.add_subplot(1, 2, 2)
    # dark.imshow(dark_channel, cmap='gray')
    # plt.show()

    return dark_channel



darkchannelcomputation(datasets.face(), 15)