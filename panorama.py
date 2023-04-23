
import math
import numpy as np
import random


from utils import pad, unpad

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

def fit_transform_matrix(p0, p1):
    """ Calcul la matrice de transformation H tel que p0 * H.T = p1

    Indication importante :
        Vous pouvez utiliser la fonction "np.linalg.lstsq" ou
        la fonction "np.linalg.svd" pour résoudre le problème.

    Entrées :
        p0 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points à transformer
        p1 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points destination

             Chaque coordonnée [x,y] dans p0 ou p1 indique 
             la position d'un point-clé [col, ligne] dans 
             l'image associée. C-à-d.  p0[i,:] = [x_i, y_i] 
             et  p1[j,:] = [x'_j, y'_j]

    Sortie :
        H  : Tableau numpy de dimension (3,3) représentant la 
             matrice de transformation d'homographie.
    """

    assert (p1.shape[0] == p0.shape[0]),\
        'Nombre différent de points en p1 et p2'

    H = None
    
    #TODO 1 : Calculez la matrice de transformation H.
    # TODO-BLOC-DEBUT    
    #raise NotImplementedError("TODO 1 : dans panorama.py non implémenté")

    
    p0_norm = np.copy(p0)
    p1_norm = np.copy(p1)
    B0 = np.eye(3)
    B1 = np.eye(3)
    s0 = np.sqrt(2) / np.mean(np.sqrt(np.sum((p0 - np.mean(p0, axis=0))**2, axis=1)))
    s1 = np.sqrt(2) / np.mean(np.sqrt(np.sum((p1 - np.mean(p1, axis=0))**2, axis=1)))
    B0[0:2, 0:2] *= s0
    B1[0:2, 0:2] *= s1
    B0[0:2, 2] = -np.mean(p0, axis=0) * s0
    B1[0:2, 2] = -np.mean(p1, axis=0) * s1
    
    p0_one = np.hstack((p0, np.ones((len(p0), 1))))
    p0_nrm = np.dot(p0_one, B0.T)[:, :2]

    p1_one = np.hstack((p1, np.ones((len(p1), 1))))
    p1_nrm = np.dot(p1_one, B1.T)[:, :2]

    N = len(p0)
    R = np.zeros((2*N, 9))

    x, y = p0_nrm[:, 0], p0_nrm[:, 1]
    u, v = p1_nrm[:, 0], p1_nrm[:, 1]

    R[::2] = np.column_stack([x, y, np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), -u*x, -u*y, -u])
    R[1::2] = np.column_stack([np.zeros(N), np.zeros(N), np.zeros(N), x, y, np.ones(N), -v*x, -v*y, -v])

    RTR = R.T @ R
    w, v = np.linalg.eig(RTR)
    H_nrm = v[:, np.argmin(w)].reshape((3, 3))

    H = np.matmul(np.linalg.inv(B1), np.matmul(H_nrm, B0))
    
    H = H / H[-1, -1]

    
    # TODO-BLOC-FIN

    return H

def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=1):
    """
    Utilize RANSAC to find a robust projective transformation

        1. Select a random set of correspondences
        2. Compute the transformation matrix 
        3. Compute the inlier correspondences
        4. Keep the largest set of inlier correspondences
        5. Finally, recompute the transformation matrix H on the
           whole set of inlier correspondences

    Inputs:
        keypoints1 -- matrix M1 x 2, each row contains the coordinates 
                      of a keypoint (x_i,y_i) in image1
        keypoints2 -- matrix M2 x 2, each row contains the coordinates 
                      of a keypoint (x'_i,y'_i) in image2
        matches  -- matrix N x 2, each row represents a correspondence
                    [index_in_keypoints1, index_in_keypoints2]
        n_iters -- the number of iterations to perform for RANSAC
        threshold -- the threshold for selecting inlier correspondences

    Outputs:
        H -- a robust estimation of the transformation matrix H
        matches[max_inliers] -- matrix (max_inliers x 2) of the inlier correspondences 
    """
    # indices of the inlier correspondences in the 'matches' array
    max_inliers = []
    
    # Homography transformation matrix
    H = None
    
    # Initialization of the random number generator
    # set the seed to compare the result returned by
    # this function with the reference solution
    random.seed(131)
    
    for i in range(n_iters):
  
        subset_indices = random.sample(range(matches.shape[0]), 4)
        indices = matches[subset_indices, :]
    

        X= keypoints1[indices[:, 0]]
        Y= keypoints2[indices[:, 1]]
        
        p0_norm = np.copy(X)
        p1_norm = np.copy(Y)
        B0 = np.eye(3)
        B1 = np.eye(3)
        s0 = np.sqrt(2) / np.mean(np.sqrt(np.sum((X - np.mean(X, axis=0))**2, axis=1)))
        s1 = np.sqrt(2) / np.mean(np.sqrt(np.sum((Y - np.mean(Y, axis=0))**2, axis=1)))
        B0[0:2, 0:2] *= s0
        B1[0:2, 0:2] *= s1
        B0[0:2, 2] = -np.mean(X, axis=0) * s0
        B1[0:2, 2] = -np.mean(Y, axis=0) * s1

        p0_one = np.hstack((X, np.ones((len(X), 1))))
        p0_nrm = np.dot(p0_one, B0.T)[:, :2]

        p1_one = np.hstack((Y, np.ones((len(Y), 1))))
        p1_nrm = np.dot(p1_one, B1.T)[:, :2]

        N = len(X)
        R = np.zeros((2*N, 9))

        x, y = p0_nrm[:, 0], p0_nrm[:, 1]
        u, v = p1_nrm[:, 0], p1_nrm[:, 1]

        R[::2] = np.column_stack([x, y, np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), -u*x, -u*y, -u])
        R[1::2] = np.column_stack([np.zeros(N), np.zeros(N), np.zeros(N), x, y, np.ones(N), -v*x, -v*y, -v])

        RTR = R.T @ R
        w, v = np.linalg.eig(RTR)
        H_nrm = v[:, np.argmin(w)].reshape((3, 3))

        H_n= np.matmul(np.linalg.inv(B1), np.matmul(H_nrm, B0))
    
        H_n = H_n / H_n[-1, -1]

        distances = np.abs(np.dot(np.hstack((keypoints1[matches[:, 0]], np.ones((matches.shape[0], 1)))), H_n.T)
                            - np.hstack((keypoints2[matches[:, 1]], np.ones((matches.shape[0], 1)))))[:, :2]
        inliers = np.where(np.linalg.norm(distances, axis=1) < threshold)[0]


        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            H = H_n
    X1 = keypoints1[matches[max_inliers, 0]]
    Y1 = keypoints2[matches[max_inliers, 1]]
    p0_norm1 = np.copy(X1)
    p1_norm1 = np.copy(Y1)
    B01 = np.eye(3)
    B11 = np.eye(3)
    s01 = np.sqrt(2) / np.mean(np.sqrt(np.sum((X1 - np.mean(X1, axis=0))**2, axis=1)))
    s11 = np.sqrt(2) / np.mean(np.sqrt(np.sum((Y1 - np.mean(Y1, axis=0))**2, axis=1)))
    B01[0:2, 0:2] *= s01
    B11[0:2, 0:2] *= s11
    B01[0:2, 2] = -np.mean(X1, axis=0) * s01
    B11[0:2, 2] = -np.mean(Y1, axis=0) * s11

    p0_one1 = np.hstack((X1, np.ones((len(X1), 1))))
    p0_nrm1 = np.dot(p0_one1, B01.T)[:, :2]

    p1_one1 = np.hstack((Y1, np.ones((len(Y1), 1))))
    p1_nrm1 = np.dot(p1_one1, B11.T)[:, :2]

    N1 = len(X1)
    R1 = np.zeros((2*N1, 9))

    x1, y1 = p0_nrm1[:, 0], p0_nrm1[:, 1]
    u1, v1 = p1_nrm1[:, 0], p1_nrm1[:, 1]

    R1[::2] = np.column_stack([x1, y1, np.ones(N1), np.zeros(N1), np.zeros(N1), np.zeros(N1), -u1*x1, -u1*y1, -u1])
    R1[1::2] = np.column_stack([np.zeros(N1), np.zeros(N1), np.zeros(N1), x1, y1, np.ones(N1), -v1*x1, -v1*y1, -v1])

    RTR1 = R1.T @ R1
    w1, v1 = np.linalg.eig(RTR1)
    H_nrm1 = v1[:, np.argmin(w1)].reshape((3, 3))

    H = np.matmul(np.linalg.inv(B11), np.matmul(H_nrm1, B01))
    
    H = H / H[-1, -1]    

    
    return H, matches[max_inliers]

def get_output_space(imgs, transforms):
    """
    Ceci est une fonction auxiliaire qui prend en entrée une liste d'images et
    des transformations associées et calcule en sortie le cadre englobant
    les images transformées.

    Entrées :
        imgs -- liste des images à transformer
        transforms -- liste des matrices de transformation.

    Sorties :
        output_shape (tuple) -- cadre englobant les images transformées.
        offset -- un tableau numpy contenant les coordonnées du coin (0,0) du cadre
    """

    assert (len(imgs) == len(transforms)),\
        'le nombre d\'images et le nombre de transformations associées ne concordent pas'

    output_shape = None
    offset = None

    # liste pour récupérer les coordonnées de tous les coins dans toutes les images
    all_corners = []

    for img, H in zip(imgs, transforms):
        # coordonnées du coin organisées en (x,y)
        r, c, _ = img.shape        
        corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])

        # transformation homographique des coins          
        warped_corners = pad(corners.astype(float)).dot(H.T).T        
        all_corners.append( unpad( np.divide(warped_corners, warped_corners[2,:] ).T ) )
                          
    # Trouver l'étendue des cadres transformées
    # La forme globale du cadre sera max - min
    all_corners = np.vstack(all_corners)

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    
    # dimension (largeur, longueur) de la zone d'affichage retournée
    output_shape = corner_max - corner_min
    
    # Conversion en nombres entiers avec np.ceil et dtype
    output_shape = tuple( np.ceil(output_shape).astype(int) )
    
    # Calcul de l'offset (horz, vert) du coin inférieur du cadre par rapport à l'origine (0,0).
    offset = corner_min

    return output_shape, offset



    
  