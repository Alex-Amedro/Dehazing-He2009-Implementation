import numpy as np

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



