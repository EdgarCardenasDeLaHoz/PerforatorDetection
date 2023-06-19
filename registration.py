import numpy as np 
import pandas as pd 

from scipy import ndimage as ndi
from skimage.measure import label
from skimage.measure import regionprops_table as r_props
from skimage import transform
from skimage.registration import phase_cross_correlation

from sklearn.neighbors import NearestNeighbors

def get_shifts(seg, n=2):

    shifts,errors = [],[]

    shifts.append(np.array((0,0)))
    errors.append(0)      

    for i in range(len(seg)-1):
        
        a = seg[i,0,::n,::n]
        b = seg[i+1,0,::n,::n]
        s,e,_ = phase_cross_correlation(a,b)

        errors.append(e)
        shifts.append(s)

    shifts = np.array(shifts)*n    
    
    return shifts, errors

def apply_registration(seg,shifts=None):
    
    if shifts is None:
        shifts = get_shifts(seg)    
        
    cum_shifts = np.cumsum(shifts,axis=0)
    seg2 = seg.copy()

    for n,shift in enumerate(cum_shifts):
        tform = transform.SimilarityTransform(scale=1, rotation=0, translation=-shift[::-1])
        s = transform.warp( seg[n,0], tform)
        if seg.dtype==np.uint8:    s = s * 255
        seg2[n,0] = s
        
    return seg2 