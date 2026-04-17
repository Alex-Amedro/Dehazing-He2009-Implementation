import numpy as np
import matplotlib.pyplot as plt
from transmission_estimation import transmission_estimation as te
def soft_matting(I : np.ndarray, t : np.ndarray ) -> np.ndarray:
    
    # Vectorisation
    H, W, C = I.shape
    N = H*W
    e = 0.0001
    I_flat = I.reshape(N, C)
    t_flat = t.flatten()
    I_pad = np.pad(I,1,mode = "edge")
    L = np.zeros((N,N))
    U = np.identity(N)
    for i in range(0,H) : 
        for j in range(0,W) :
            voisinage = I[max(0 ,i-1):min(H-1,i+2),max(0,j-1):min(W-1,j+2)]
            moyenne_int = np.mean(voisinage, axis=0)
            moyenne_k = np.mean(moyenne_int, axis = 0)
            voinisage_flat = voisinage.reshape(-1,3)
            cov_k = np.cov(voinisage_flat, rowvar=False, ddof=0)
            for l in range(9):
                for k in range(9):
                    kron = 0
                    x_l = l % 3 - 1
                    y_l = l // 3 - 1 

                    x_k = k % 3 - 1
                    y_k = k // 3 -1 

                    idx_l = (i-x_l)*(j-y_l)
                    idx_k = (i-x_l)*(j-y_l)
                    if idx_l == idx_k :
                        kron = 1
                    A = np.transpose(I(idx_l)-moyenne_k)
                    B = np.linalg.inv(cov_k+(e/9)*(U))
                    C = I(idx_k)-moyenne_k
                    L[idx_l][idx_k] += kron - ((1/9)*(1+np.matmul(np.matmul(A,B),C)))



if __name__ == "__main__":
    img = plt.imread("../../data/raw_images/image1.jpg")
    transminsion = te(img, 50)
    soft_matting(img,transminsion)
    