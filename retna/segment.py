import glob 
import matplotlib.pyplot as plt
import cv2 
import torch
import numpy as np
from skimage.measure import find_contours, approximate_polygon
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
from skimage import morphology as mor

def segment_data(model,data):
    frames = []   
    frame_nums = np.linspace(0,len(data)-1,50).astype(int)
    
    for n in frame_nums:
        print(".",end="")
        
        d = data[n]   
        d = d - np.min(d)
        im = d / np.max(d) 
        
        im_in = im[:,:,None]*255
        pred = model.segment(im_in,scale=0.8)
    
        pred[pred<0.5] = 0
        
        draw = draw_panels(im, pred)

        frames.append(draw)
    return frames

######################################
def draw_panels(im, pred):
    
    imin_3 = im[...,None]*[1,1,1]*255
    imin_3 = imin_3.astype(np.uint8)
    
    lb_rgb = prediction_to_rgb(pred, imin_3)
    shapes = prediction_to_shapes(pred)
    anno_3 = draw_on_im( imin_3, shapes)

    #label = ndi.label(pred[1]>0.45)[0]
    #shapes = label_to_shapes(label, kind="poly", tol=3)
    #labl_3 = draw_on_im( imin_3, shapes)

    draw = np.hstack([imin_3, lb_rgb, anno_3])    
    
    return draw

#######################################
def prediction_to_rgb(pred, imin_3):

    remap = np.array([[1,0,0],[0,1,0]])

    lb_rgb = np.dot(pred.transpose([1,2,0]),remap)
    lb_rgb2 = imin_3*(1-lb_rgb) + lb_rgb*255
    
    lb_rgb2 =lb_rgb2.astype(np.uint8)
    #lb_rgb = (lb_rgb*255).astype(np.uint8)

    return lb_rgb2

def prediction_to_shapes(pred):

    shapes = []

    colors = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]])*255
    tol = [5,1]

    for n,prd in enumerate(pred):

        bP = prd>0.45
        clr = tuple(colors[n])
        shps = binary_to_shape(bP,kind="poly",color=clr,tol=tol[n])
        shapes.extend(shps)

    return shapes

#######################################
def label_to_shapes(label, kind="poly", tol=3):
    
    colors = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]])*255
    shapes = []
    if kind is "poly":

        bI = label>0
        # mes.regionprops(label props =[perimeter] )
        for n,contour in enumerate(find_contours(bI, 0)):

            idx = n % 6 
            clr = tuple(colors[idx])

            coords = approximate_polygon(contour, tolerance=tol)
            coords = coords[:,[1,0]].ravel()
            shp = { "type":kind, "xy":tuple(coords), "color":clr }
            shapes.append(shp)
            
    if kind is "rect":

        rprops = mes.regionprops(label)
        for n, r in enumerate(rprops) :

            idx = n % 6 
            clr = tuple(colors[idx])

            xy = tuple(np.array(r.bbox)[[1,0,3,2]])
            shp = {"type":kind,"xy":xy, "color":clr}
            shapes.append(shp)

    return shapes

def binary_to_shape(bI,kind="poly", color="red", tol=3):
    
    shapes = []
    if kind is "poly":
        for contour in find_contours(bI, 0):
            coords = approximate_polygon(contour, tolerance=tol)
            coords = coords[:,[1,0]].ravel()
            shp = {"type":kind,"xy":tuple(coords), "color":color}
            shapes.append(shp)
            
    if kind is "rect":
        for r in mes.regionprops(mes.label(bI)) :
            xy = tuple(np.array(r.bbox)[[1,0,3,2]])
            shp = {"type":kind,"xy":xy, "color":color}
            shapes.append(shp)
    return shapes

########################################

def draw_on_im(im, shape):
    
    p_im = Image.fromarray(im, 'RGB') 
    draw = ImageDraw.Draw(p_im)  
    for shp in shape:  
        color = shp["color"]
        if shp["type"] is "rect":
            draw.rectangle(shp["xy"], fill=None, outline=color, width=3)   
        if shp["type"] is "poly":
            draw.line(shp["xy"], fill=color, width=3)
    im_out = np.array(p_im)
    return im_out    

def render_frames(outname, frame_list ):

    size = np.array(frame_list[0].shape)[[1,0]] * 0.75
    size = tuple(size.astype(int))
    fcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(outname,fcc, 5, size ) 
    for frame in frame_list:
            frame = cv2.resize(frame,size)
            out_vid.write(frame)
    out_vid.release()