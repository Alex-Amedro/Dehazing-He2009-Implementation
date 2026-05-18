import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse
import os
from image_reconstruction import reconstruction as ir 
from soft_matting import soft_matting as sm
from transmission_estimation import transmission_estimation as te
from cache_utils import load_cache, save_cache, get_img_name

def depth_map(I, patch_size : int = 50, omega : float = 0.95,
              lambda_e : float = 0.0001, img_name : str = None):

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

    d = -np.log(t)
    return d 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcul de la carte de profondeur.")
    parser.add_argument("image", type=str, help="Chemin vers l'image d'entrée")
    parser.add_argument("--patch-size", type=int, default=20, help="Taille du patch pour le dark channel (défaut: 20)")
    parser.add_argument("--omega", type=float, default=0.95, help="Coefficient omega pour l'estimation de transmission (défaut: 0.95)")
    parser.add_argument("--lambda-e", type=float, default=0.0001, help="Paramètre de régularisation lambda du soft matting (défaut: 0.0001)")
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    img_name = get_img_name(img_path)
    
    img = plt.imread(img_path)
    reconstructed_image = depth_map(img, patch_size=args.patch_size, omega=args.omega,
                                    lambda_e=args.lambda_e, img_name=img_name)
    
    fig = plt.figure()
    original = fig.add_subplot(1, 2, 1)
    original.imshow(img)
    original.set_title("Image originale")
    original.axis('off')
    
    reconstructed = fig.add_subplot(1, 2, 2)
    reconstructed.imshow(reconstructed_image)
    reconstructed.set_title("Depth Map")
    reconstructed.axis('off')
    
    plt.show()