
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects
from skimage import measure

def get_features(b_I):
    label_image = measure.label(b_I)
    feat = pd.DataFrame( measure.regionprops_table(label_image, 
                properties=['label', 'area', 'centroid']) )
    return feat
    
def knn_edges(src, dst, n_neigh=5):
  
    neigh = NearestNeighbors(n_neighbors=n_neigh).fit(dst)
    dist, inds_B = neigh.kneighbors(src, return_distance=True)
    inds_A = np.tile(range(inds_B.shape[0]),(n_neigh,1)).T
    edges = np.vstack((inds_A.ravel(),inds_B.ravel(),dist.ravel())).T
    
    return edges 

def greedy_assignment(edges):

    sorted_edges = edges[np.argsort(edges[:,2]),:]

    A_taken, B_taken = [],[]
    assign_edges = []
    
    for A,B,dist in sorted_edges:

        if (A not in A_taken) and (B not in B_taken):
            assign_edges.append([A,B,dist]) 
            A_taken.append(A)
            B_taken.append(B)
            
    assign_edges = np.array(assign_edges)    
    return assign_edges 

def pair_pts(src,dst):
    
    edges = knn_edges(src, dst)
    edges = greedy_assignment(edges)
    
    return edges 

def get_matched_area(df_targ,df_pred):

    cent_t = np.array(df_targ[["centroid-0","centroid-1"]])
    area_t = np.array(df_targ[["area"]])

    cent_p = np.array(df_pred[["centroid-0","centroid-1"]])
    area_p = np.array(df_pred[["area"]])

    edges = pair_pts(cent_t,cent_p)
    edges = edges[edges[:,2]<5,:]

    X = area_t[edges[:,0].astype(np.int)]
    Y = area_p[edges[:,1].astype(np.int)]
    
    return X,Y
