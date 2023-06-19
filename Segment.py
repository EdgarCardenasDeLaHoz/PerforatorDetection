import retna
import numpy as np

import glob as glob

from . import im_proc
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import ndimage as ndi

def label2rgb(L):
    if L.dtype == bool: L,_ = ndi.label(L)
    nC = 10
    L1 = L % nC
    cmap = cm.get_cmap('rainbow',nC)
    dL = cmap(L1)[:,:,:3]
    dL[L==0,:] = 0
    dL = (dL * 255).astype(np.uint8)
    
    return dL

def label_nearest(label, radius=10, blockSeeds = True):
    
    if label.dtype == bool:
        bI = label
        label,_ = ndi.label(bI)
    else:
        bI = label > 0

    ##
    distances, nearest_label_coords = ndi.distance_transform_cdt(
    label == 0, return_indices=True  )

    labels_out = np.zeros_like(label)
    dilate_mask = distances <= radius
   
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels

    if blockSeeds:
        labels_out[bI]=0

    return labels_out    


def label_preop(im, tx=-1):

    w = 5
    im0 = im[0:w].mean(axis=0)
    imx = im[(tx-w):(tx)].mean(axis=0)  
    dPred = 1.*imx - im0

    ###############################

    ############
    bFlap = label_flap(im0, dPred)
    bFlap2 = im_proc.bi_erode(bFlap,(5),.3) 
    if bFlap2.sum()==0: return np.zeros((4,) + im0.shape)
    bNavel = label_navel(im0, dPred, bFlap2)
    bFlap2 = bFlap2 & ~bNavel
    bPerf  = label_pref(imx,dPred,bFlap2)
    bFlap2  = bFlap2 & ~bPerf
    bFalse = label_false(im0,dPred,bFlap2)

    classes = np.array([bFlap ,bNavel,bPerf,bFalse])
    
    return classes
    
def measure_preop(im,classes):
    colors = "rgbk" 
    measurements = np.zeros((len(im),len(classes)))
    for t in range(0,len(im)):  
        for n,c in enumerate(classes):

            x = im[t][c].mean()
            measurements[t,n] = x
    
    return measurements
    
def label_flap(im0, dPred):
    
    bFlap = im0 < np.percentile(im0,40) 
    #bFlap = bFlap & (dPred>10)
    bFlap = im_proc.area_filter(bFlap,100)>0
    bFlap = im_proc.bi_close(bFlap,10,.5)
    bFlap = im_proc.bi_open(bFlap,5,.5)
    bFlap = im_proc.get_largest(bFlap)>0
    bFlap = im_proc.area_filter(bFlap==0,5000)==0
    
    return bFlap
    
def label_navel(im0, dPred, bFlap):   
    
    x = im0[bFlap]   

    bFlap = 1.*bFlap
    bFlap[:,-150:] = 0 
    bFlap[:,:150] = 0 

    nav_thr = np.percentile(x,95) #75
    bNav = (im0 > nav_thr) 
    bNav = bNav & (dPred<25)
    bNav = bNav & (bFlap>0)


    bNav = im_proc.area_filter(bNav,10,5000)>0
    bNav = im_proc.get_largest(bNav)>0
    
    return bNav

def label_pref(imx,dPred,bFlap):
    
    x = imx[bFlap]

    pref_thr = np.median(x)+1*x.std() #90
    x = dPred[bFlap]
    dpref_thr = np.median(x)+1*x.std() #90
    bPref = imx > pref_thr
    bPref = bPref & (dPred>dpref_thr)
    bPref = bPref & bFlap
    bPref = im_proc.area_filter(bPref,50,5000)>0
    
    return bPref
    
def label_false(im0,dPred,bFlap):
    x = im0[bFlap]
    dx = dPred[bFlap]
    mis_thr = x.mean()+0.8*x.std() #45
    dmis_thr = dx.mean()-1*dx.std() #55
    bMiss = (im0 > mis_thr) & (dPred<dmis_thr)
    bMiss = bMiss & bFlap
    bMiss = im_proc.area_filter(bMiss,50)>0
    
    return bMiss

def label_nearest(label):
    if label.dtype == np.bool:
        bI = label>0
        label,_ = ndi.label( bI )
    else:
        bI = label > 0

    dists, nrst_L_coors = ndi.distance_transform_cdt(
    bI, return_indices=True  )

    label_out = np.zeros_like(label)
    dilate_mask = dists <= radius
   
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels

    return labels_out

############################################# 
#  Plotting 
# ######################################   

def draw_labels(im, classes, ax=None):

    tx = -1
    im0 = im[0:5].mean(axis=0)
    imx = im[(tx-10):(tx-5)].mean(axis=0)  


    dPred = 1.*imx - im0
    
    #d_labels = labels/np.max(labels) * 255
    idx = classes[0]*0
    for n,cls in enumerate(classes):
        idx[cls]=n+1

    colors = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0.5,1]])*255
    d_labels = colors[idx]

    draw1 = np.hstack((im0,imx,))
    draw1 = np.tile(draw1[:,:,None],[1,1,3])
    dPred = np.tile(dPred[:,:,None],[1,1,3])

    draw2 = np.hstack((dPred+125,d_labels)) 
    draw = np.vstack((draw1,draw2))
    draw = draw.astype(np.uint8)

    if ax is None:
        ax = plt.axes()

    ax.imshow(draw, cmap="jet")
    ax.axis("off")    
#################################################    
def plot_measurements(measurements):
    plt.figure()
    for m in measurements:  
        meas = m - m[:,0:1]
        meas = meas / meas[0,1]
        for n,c in enumerate(range(meas.shape[1])): 
            clr = colors[n]
            x = meas[:,n]
            tx = range(len(x))
            plt.plot(tx, x, "-o", color=clr)
            

def draw_crosssection(im,x=250,y=150,t=-5, cmap="jet"):
    plt.close("all")
    fig = plt.figure()
            
    ax = plt.subplot2grid((3,3), (0,0), rowspan=2, colspan=2)
    ax.imshow(im[t],cmap=cmap)
    ax = plt.subplot2grid((3,3), (0,2) ,rowspan=2)
    ax.imshow(im[:,::2,x].T, cmap=cmap)
    ax = plt.subplot2grid((3,3), (2,0), colspan=2)
    ax.imshow(im[:,y,::2], cmap=cmap)
    
    plt.subplots_adjust(left=0.05, bottom=0.04, 
        right=0.99, top=0.99, wspace = 0.0, hspace=0)   

    #plt.tight_layout()
def dice_loss(pred, target, smooth = .0001):

    if pred.ndim == 3: pred = pred[None,:]
    if target.ndim == 3: target = target[None,:]

    if pred.shape[1] != target.shape[1]:
        print("loss warning - Shapes are different sizes",
                pred.shape[1],target.shape[1])

    intersection = 2*(pred * target).mean(axis=(2,3))
    combination =  (pred**2 + target**2).mean(axis=(2,3))
    dsc = (intersection + smooth) / (combination+smooth) 
    dsc = (1 - dsc)
    return dsc   

 
    
def render_frames(outname, frame_list , fps=5):

    size = np.array(frame_list[0].shape)[[1,0]] 
    size = tuple(size.astype(int))
    fcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(outname,fcc, fps, size ) 
    for frame in frame_list:
        frame = cv2.resize(frame,size)
        out_vid.write(frame[:,:,::-1])

    out_vid.release()