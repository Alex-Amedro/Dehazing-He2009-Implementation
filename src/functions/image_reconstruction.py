import numpy as np
import matplotlib.pyplot as plt
import os
from soft_matting import soft_matting as sm
from transmission_estimation import transmission_estimation as te
from cache_utils import load_cache, save_cache, get_img_name

def reconstruction(I : np.ndarray, img_name : str = None):
    # Sécurité : conversion en float et normalisation entre 0 et 1
    # indispensable pour éviter l'underflow et correspondre au clip final.
    if I.dtype == np.uint8:
        I = I.astype(np.float64) / 255.0

    t_tild, A = te(I, 50, img_name=img_name)

    # Vérifier le cache pour le soft matting
    if img_name is not None:
        cached = load_cache(img_name, 'soft_matting')
        if cached is not None:
            t = cached
        else:
            t = sm(I, t_tild, img_name)
    else:
        t = sm(I, t_tild)

    t0 = 0.1

    # On contraint la transmission à t0 dans les zones très denses
    t_bounded = np.maximum(t, t0)

    # Utilisation de np.newaxis pour le broadcasting : (H, W) devient (H, W, 1)
    # ce qui permet la division avec I qui est de dimension (H, W, 3)
    J = (I - A) / t_bounded[..., np.newaxis] + A

    return J, t

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/raw_images/temple.png")
    img_name = get_img_name(img_path)

    img = plt.imread(img_path)
    reconstructed_image, transmission = reconstruction(img, img_name)

    fig = plt.figure()
    original = fig.add_subplot(1, 3, 1)
    original.imshow(img)
    original.set_title("Image originale")
    original.axis('off')

    trans_plot = fig.add_subplot(1, 3, 2)
    trans_plot.imshow(transmission, cmap='gray')
    trans_plot.set_title("Transmission")
    trans_plot.axis('off')

    reconstructed = fig.add_subplot(1, 3, 3)
    reconstructed.imshow(reconstructed_image)
    reconstructed.set_title("Sans brume")
    reconstructed.axis('off')

    plt.show()