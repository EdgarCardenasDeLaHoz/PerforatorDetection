import numpy as np 
import pandas as pd 

import cv2
import torch
import matplotlib.pyplot as plt 

from scipy import ndimage as ndi
from scipy.ndimage.filters import generic_filter

from skimage import filters
from skimage.measure import regionprops_table as r_props
from skimage import morphology as mor
from skimage.segmentation import clear_border

def bi_open( bI,radius,down_sample=0.2):
    
    im_size = np.array( bI.shape )
    d_size = tuple((im_size*down_sample).astype(np.int))
    im_size = tuple(im_size)

    filt = cv2.resize((bI*1).astype(np.uint8), d_size[::-1]) > 0.3
    dsk = mor.disk(radius)
    filt = ndi.binary_opening( filt, dsk)
    bI_out = cv2.resize((filt*1).astype(np.uint8), im_size[::-1])>0.3

    return bI_out

def bi_erode( bI,radius,down_sample=0.2):
    
    im_size = np.array( bI.shape )
    d_size = tuple((im_size*down_sample).astype(np.int))
    im_size = tuple(im_size)

    filt = cv2.resize((bI*1).astype(np.uint8), d_size[::-1]) > 0.2
    dsk = mor.disk(radius)
    filt = ndi.binary_erosion( filt, dsk)
    bI_out = cv2.resize((filt*1).astype(np.uint8), im_size[::-1])>0

    return bI_out

def bi_close(  bI,radius,down_sample=0.2):
    
    im_size = np.array( bI.shape )
    d_size = tuple((im_size*down_sample).astype(np.int))
    im_size = tuple(im_size)

    filt = cv2.resize((bI*1).astype(np.uint8), d_size[::-1]) > 0.2
    dsk = mor.disk(radius)
    filt = ndi.binary_closing( filt, dsk)
    bI_out = cv2.resize((filt*1).astype(np.uint8), im_size[::-1])>0.5

    return bI_out

def bi_dilate( bI,radius,down_sample=0.5):

    im_size = np.array( bI.shape )
    d_size = tuple((im_size*down_sample).astype(np.int))
    im_size = tuple(im_size)

    filt = cv2.resize((bI*1.), d_size[::-1]) > 0
    filt = ndi.binary_dilation( filt, mor.disk(radius) )
    bI_out = cv2.resize((filt*1).astype(np.uint8), im_size[::-1])>0.5

    return bI_out

def back_filt(im, sigma, down_sample=0.5):

    im_size = np.array(im.shape)
    d_size = (im_size*down_sample).astype(np.int)

    filt = cv2.resize(im, tuple(d_size[::-1]))
    filt = ndi.minimum_filter(filt, size=sigma)
    filt = filters.gaussian( filt , sigma=sigma, truncate=3)

    im_out = cv2.resize(filt, tuple(im_size[::-1]))>0.5

    return im_out

def area_filter(ar, min_size=0, max_size=None):
    """
    """
    if ar.dtype == bool:
        ccs,l_max = ndi.label(ar)
    else:
        ccs = ar
        l_max = ar.max()

    component_sizes = np.bincount(ccs[ccs>0])

    idxs = np.arange(l_max+1).astype(np.uint16)
    if min_size>0:
        too_small = component_sizes < min_size
        idxs[too_small]=0

    if max_size is not None:
        too_large = component_sizes > max_size
        idxs[too_large]=0

    out = np.zeros_like(ccs, np.uint16)
    _, idxs2 = np.unique(idxs,return_inverse=True)
    out[ccs>0] = idxs2[ccs[ccs>0]]

    return out

def get_largest(ar):
    """
    """
    if ar.dtype == bool:
        ccs,l_max = ndi.label(ar)
    else:
        ccs = ar
        l_max = ar.max()
    
    if np.sum(ccs)>0: 
        component_sizes = np.bincount(ccs[ccs>0])
        out = ccs == (np.argmax(component_sizes))
    else:
        out = (ccs*0)>0

    return out
