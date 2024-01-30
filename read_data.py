import h5py 
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

class H5Reader():
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(filename, 'r') as fh:
            self.shape = fh["im_data"].shape

    def __len__(self):
        return self.shape[0]
        
    def __getitem__(self, frame_num):
          return read_H5(self.filename, frame_num)

class XviReader():
    def __init__(self, filename):
        self.filename = filename

        size = os.path.getsize(filename)
        n_frames = int((size-1024)/(32+640*480*2))-1
        n_frames = max((n_frames,0))
        self.shape = (n_frames, 480, 640 )

    def __len__(self):
        return self.shape[0]
        
    def __getitem__(self, frame_num):
          return read_xvi(self.filename, frame_num)[0]

############################
##     Helper functions 
############################

def read_file(filename):

    ext = filename.split(".")[-1]

    if ext.lower() == "h5":
        data = H5Reader(filename)

    elif ext.lower() == "xvi":
        data = XviReader(filename)

    return data

############################
##       IO HDF5
###########################

def read_H5(fn,t=None):
    with h5py.File(fn, 'r') as fh:

        attributes = {}
        for k in fh["im_data"].attrs.keys(): attributes[k] = fh["im_data"].attrs[k]           

        if t is None:
            im_data = fh["im_data"][:]
        else:
            if len(fh["im_data"]) < t: return None,attributes
            
            im_data = fh["im_data"][t]
            return im_data, attributes

    im_data = np.array(im_data)
    return im_data, attributes

def save_h5(fn, im, attributes=None):  
    
    with h5py.File(fn, 'a') as fh:

        if "im_data" in fh.keys(): del fh["im_data"]

        dset = fh.create_dataset("im_data", data=im , compression="lzf" )

        if attributes is not None:
            for key in attributes:
                dset.attrs[key] = attributes[key]

###########################
##       IO XVI
###########################

def read_xvi(filename, frame_num=None):

    filename.lower().endswith(('.xvi'))

    size = os.path.getsize(filename)
    n_frames = int((size-1024)/(32+640*480*2))

    infile = open(filename, 'rb')

    if frame_num is None:
        frames = [ read_xvi_fr(infile, n) for n in range(1,n_frames)  ]
        xviFrame = np.array(frames)
    else:
        n = frame_num+1
        xviFrame = read_xvi_fr(infile, n)
    
    xviFrame = xviFrame.reshape(-1,480, 640).astype(np.uint16)

    return xviFrame 

def read_xvi_fr(infile, frame_num):

    n = int(frame_num)
    pos = 1024 + (32+640*480*2) *n
    infile.seek(pos)
    buf = infile.read(640*480*2)    
    xviFrame = np.frombuffer(buf, dtype='uint16')

    return xviFrame

def get_num_frames(filename):

    size = os.path.getsize(filename)
    n_frames = int((size-1024)/(32+640*480*2))-1
    n_frames = max((n_frames,0))

    return n_frames


    
