
import numpy as np 
from skimage import io, transform


def calc_loss(pred, target, method=None, select_channel=None):

    if select_channel is not None:
        target = target[:,select_channel]
        pred = pred[:,select_channel]

    if method == "any":
        max_targ,_ = target.max(dim=1,keepdim=True)
        loss = dice_loss(pred, max_targ)
    else:
        loss = dice_loss(pred, target)

    loss[:,0] = 0
    loss[:,1] = loss[:,1]*0.5
    loss[:,2] = loss[:,2]*2

    loss = loss.mean()       
    return loss

def dice_loss(pred, target, smooth = .00000000001):

    if pred.ndim == 3: pred = pred[None,:]
    if target.ndim == 3: target = target[None,:]

    if pred.shape[1] != target.shape[1]:
        print("loss warning - Shapes are different sizes",
                pred.shape[1],target.shape[1])

    intersection = 2*(pred * target).mean(dim=(2,3))
    combination =  (pred**2 + target**2).mean(dim=(2,3))
    dsc = (intersection + smooth) / (combination+smooth) 
    dsc = (1 - dsc)
    return dsc

