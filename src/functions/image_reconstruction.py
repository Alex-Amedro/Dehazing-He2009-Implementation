import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from soft_matting import soft_matting as sm
from transmission_estimation import transmission_estimation as te
from cache_utils import load_cache, save_cache, get_img_name

def reconstruction(I : np.ndarray, patch_size : int = 50, omega : float = 0.95,
                   t0 : float = 0.1, lambda_e : float = 0.0001, img_name : str = None):
    # Sécurité : conversion en float et normalisation entre 0 et 1
    # indispensable pour éviter l'underflow et correspondre au clip final.
    if I.dtype == np.uint8:
        I = I.astype(np.float64) / 255.0

    t_tild, A = te(I, patch_size, omega=omega, img_name=img_name)

    # Vérifier le cache pour le soft matting
    if img_name is not None:
        cached = load_cache(img_name, 'soft_matting')
        if cached is not None:
            t = cached
        else:
            t = sm(I, t_tild, img_name, lambda_e=lambda_e)
    else:
        t = sm(I, t_tild, lambda_e=lambda_e)

    # On contraint la transmission à t0 dans les zones très denses
    t_bounded = np.maximum(t, t0)

    # Utilisation de np.newaxis pour le broadcasting : (H, W) devient (H, W, 1)
    # ce qui permet la division avec I qui est de dimension (H, W, 3)
    J = (I - A) / t_bounded[..., np.newaxis] + A

    return J, t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction d'image sans brume (He et al. 2009).")
    parser.add_argument("image", type=str, help="Chemin vers l'image d'entrée")
    parser.add_argument("--patch-size", type=int, default=20, help="Taille du patch pour le dark channel (défaut: 20)")
    parser.add_argument("--omega", type=float, default=0.95, help="Coefficient omega pour l'estimation de transmission (défaut: 0.95)")
    parser.add_argument("--t0", type=float, default=0.1, help="Seuil minimal de transmission (défaut: 0.1)")
    parser.add_argument("--lambda-e", type=float, default=0.0001, help="Paramètre de régularisation lambda du soft matting (défaut: 0.0001)")
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    img_name = get_img_name(img_path)

    img = plt.imread(img_path)
    reconstructed_image, transmission = reconstruction(
        img, patch_size=args.patch_size, omega=args.omega,
        t0=args.t0, lambda_e=args.lambda_e, img_name=img_name
    )

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