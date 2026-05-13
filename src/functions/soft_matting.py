import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
from transmission_estimation import transmission_estimation as te

def soft_matting(I : np.ndarray, t_tild : np.ndarray ) -> np.ndarray:
    # para
    taille_fenetre = 3
    norm_w = taille_fenetre*taille_fenetre
    e = 0.0001
    lambda_e = 0.0001
    U_3 = np.identity(3)

    # Vectorisation
    H, W, C = I.shape
    N = H*W
    I_flat = I.reshape(N, C)
    t_tild_flat = t_tild.flatten()

    U = sp.sparse.eye(N, format='csr')

    #padding sur les cotés de l'image en copiant le 3 canaux couleurs
    I_pad = np.pad(I, ((1, 1), (1, 1), (0, 0)), mode = "edge")
    
    # L
    indices_lignes = []
    indices_colonnes = []
    valeurs = []

    # parcours pixels
    total_pixels = (H-3) * (W-3)
    current_pixel = 0
    for i in range(1,H-2) :
        for j in range(1,W-2) :
            current_pixel += 1
            if current_pixel % max(1, total_pixels // 20) == 0:  # Afficher tous les 5%
                progress = (current_pixel / total_pixels) * 100
                print(f"Progression: {progress:.1f}%")
            voisinage = I[i:i+taille_fenetre,j:j+taille_fenetre,:]
            moyenne_int = np.mean(voisinage, axis=0)
            moyenne_k = np.mean(moyenne_int, axis = 0)
            voinisage_flat = voisinage.reshape(-1,3)
            cov_k = np.cov(voinisage_flat, rowvar=False, ddof=0)
            B = np.linalg.inv(cov_k+(e/9)*(U_3))

            # parcours de couple dans la fenetres
            for l in range(9):
                for k in range(9):
                    kron = 0
                    x_l = l % 3 - 1
                    y_l = l // 3 - 1 

                    x_k = k % 3 - 1
                    y_k = k // 3 -1 

                    i_l = min(max(i + y_l, 0), H - 1)
                    j_l = min(max(j + x_l, 0), W - 1)
                    
                    i_k = min(max(i + y_k, 0), H - 1)
                    j_k = min(max(j + x_k, 0), W - 1)

                    idx_l = i_l * W + j_l
                    idx_k = i_k * W + j_k

                    if l == k :
                        kron = 1

                    A = voinisage_flat[l]-moyenne_k
                    C = voinisage_flat[k]-moyenne_k

                    value_mat = np.dot(A,np.dot(B,C))
                    value_L = kron - ((1/9)*(1+value_mat))
                    indices_lignes.append(idx_l)
                    indices_colonnes.append(idx_k)
                    valeurs.append(value_L)

    L = sp.sparse.coo_matrix((valeurs, (indices_lignes, indices_colonnes)), shape=(N, N)).tocsr()
    A = L + lambda_e * U
    b = lambda_e * t_tild_flat
    t_flat, _ = sp.sparse.linalg.cg(A, b)
    t = t_flat.reshape(H, W)

    np.save('/tmp/transmission_soft_matting.npy', t)
    return t

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "../../data/raw_images/image2.jpeg")
    img = plt.imread(img_path)
    transmission_bloc= te(img, 50)
    transmission = soft_matting(img,transmission_bloc)
    fig = plt.figure()
    original = fig.add_subplot(1, 3, 1)
    original.imshow(img)
    trs_tild = fig.add_subplot(1, 3, 2)
    trs_tild.imshow(transmission_bloc, cmap='gray')
    trs = fig.add_subplot(1, 3, 3)
    trs.imshow(transmission, cmap='gray')
    plt.show()
    