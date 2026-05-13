import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
from image_reconstruction import reconstruction as ir 
from soft_matting import soft_matting as sm
from transmission_estimation import transmission_estimation as te
from cache_utils import load_cache, save_cache, get_img_name

def depth_map(I, img_name : str = None):

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

    d = -np.log(t)
    return d 

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/raw_images/image2.jpeg")
    img_name = get_img_name(img_path)
    
    img = plt.imread(img_path)
    reconstructed_image = depth_map(img, img_name)
    
    fig = plt.figure()
    original = fig.add_subplot(1, 2, 1)
    original.imshow(img)
    original.set_title("Image originale")
    original.axis('off')
    
    reconstructed = fig.add_subplot(1, 2, 2)
    reconstructed.imshow(reconstructed_image)
    reconstructed.set_title("Sans brume")
    reconstructed.axis('off')
    
    plt.show()