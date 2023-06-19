import os 
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import h5py
import cv2

from .training import train_model
 
class UNet(nn.Module):

    def __init__(self, in_channels, out_classes):
        super().__init__()
        
        num_chan = [8,8,8,8]
        self.dconv_down1 = double_conv(in_channels, num_chan[0])
        self.dconv_down2 = double_conv(num_chan[0], num_chan[1])
        self.dconv_down3 = double_conv(num_chan[1], num_chan[2])
        self.dconv_down4 = double_conv(num_chan[2], num_chan[3])        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(num_chan[2] + num_chan[3], num_chan[2])
        self.dconv_up2 = double_conv(num_chan[1] + num_chan[2], num_chan[1])
        self.dconv_up1 = double_conv(num_chan[0] + num_chan[1], num_chan[0])
        self.conv_last = nn.Conv2d( num_chan[0], out_classes, 1)
        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)
        x = self.upsample(x)    

        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x) 

        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        
        return out
    
    def train_model(self, data_loader, optimizer=None, num_epochs=10):

        train_model(self, data_loader, optimizer, num_epochs)

class V_Net(nn.Module):

    def __init__(self, in_channels, out_classes, hidden_channels = None):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [8,8,8,8]

        hidden_channels.insert(0,in_channels)

        Mod_list = []
        for i in range(len(hidden_channels)-1):

            hid_chan1 = hidden_channels[i]
            hid_chan2 = hidden_channels[i+1]
            block = res_conv( hid_chan1, hid_chan2)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)
        self.maxpool = nn.AvgPool2d(2)       
        self.conv_last = nn.Conv2d( sum(hidden_channels), out_classes, 1)

        self.prescale = None
        
    def forward(self, x_in):
        
        x = x_in

        out_list = [x_in]
        for n, block in enumerate(self.blocks):

            x1 = block(x)
            x = lims_normalize(x1)
            x_out = F.interpolate(x, size=x_in.shape[2:], mode='bilinear', align_corners=False)
            out_list.append(x_out)         

        x = torch.cat(out_list, dim=1)
        
        out = self.conv_last(x)

        out = out - out.min()
        outmax = out.max()
        if outmax > 1:  out = out / outmax

        return out

    def train_model(self, data_loader, optimizer=None, num_epochs=10):
        train_model(self, data_loader, optimizer, num_epochs)

    def eval_file(self, filename ):
        return eval_file(self,filename)

    def segment(self, image, scale=None):
        if scale is None:
            if self.prescale is None: scale = 1
            else: scale = self.prescale

        return segment(self,image, scale=scale)

    def save(self, name, path = ".\\models\\"):
        if not os.path.isdir(path):   os.mkdir(path)
        fn = path + name + ".pt"
        torch.save( self, fn )

class Retna_V1(nn.Module):

    def __init__(self, in_channels, out_classes, hidden_channels = None):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [8,8,8,8]

        hidden_channels.insert(0,in_channels)

        Mod_list = []
        for i in range(len(hidden_channels)-1):
            if i == 0:
                hid_chan1 = hidden_channels[i]
            else:
                hid_chan1 = hidden_channels[i]+in_channels
            hid_chan2 = hidden_channels[i+1]
            block = res_conv( hid_chan1, hid_chan2)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)
        self.maxpool = nn.AvgPool2d(2)       
        self.conv_last = nn.Conv2d( sum(hidden_channels), out_classes, 1)
        
    def forward(self, x_in):
        
        x_next = x_in

        out_list = [x_in]
        for n, block in enumerate(self.blocks):

            x_next = block(x_next)     
            x_out = F.interpolate(x_next, size=x_in.shape[2:], mode='bilinear', align_corners=False)
            x_in2 = F.interpolate(x_in, size=x_next.shape[2:], mode='bilinear', align_corners=False)
            x_next = torch.cat( [x_in2, x_next], dim=1)

            out_list.append(x_out)         

        x = torch.cat(out_list, dim=1)
        out = self.conv_last(x)

        if self.training:
            out = F.leaky_relu(out)
            out = 1-F.leaky_relu(1-out)
        else:
            out = torch.clamp(out,0,1)

        return out

    def train_model(self, data_loader, optimizer=None, num_epochs=10):
        train_model(self, data_loader, optimizer, num_epochs)

    def eval_file(self, filename ):
        return eval_file(self,filename)

    def segment(self, image, scale=1):
        return segment(self,image, scale=scale)

    def save(self, name, path = ".\\models\\"):
        if not os.path.isdir(path):   os.mkdir(path)
        fn = path + name + ".pt"
        torch.save( self, fn )


#####################################################
##              BASE Functions                     ##
#####################################################

def eval_file(model,filename):

    X1,X2,Y1,Y2 = 148,1852,398,2022
    in_im  = np.array(cv2.imread( filename))
    out_im = np.zeros((in_im.shape[0],in_im.shape[1],1))
    t_im = in_im[X1:X2,Y1:Y2,0:1]/256

    t_im = np.transpose( t_im, [2,0,1]) 
    t_im = torch.tensor(t_im).float()[None,:]   

    model.eval()
    with model.no_grad():
        output = model(t_im)

    output = output.detach().numpy()[0,0]
    out_im[X1:X2,Y1:Y2,0] = output

    return in_im,out_im

def segment(model,image, scale=1):

    if image.ndim==4 and len(image)==1:
        image = image[0]

    if image.ndim == 2:
        image = image[...,None]

    if image.shape[0] > image.shape[-1]:
        image = np.transpose( image, [2,0,1])

    if isinstance(image, torch.Tensor):  image = image.cpu().detach().numpy()
  
    if image.dtype == np.uint8:   image = image/256.
    if image.max()>20: image = image/256.
    
    sz = np.array(image.shape[1:3])
    image = resize_to_base(image, scale)

    image = torch.tensor( image ).float()[None,:]   
    if next(model.parameters()).is_cuda: image =image.to("cuda")    

    model.eval()
    with torch.no_grad():  output = model(image)  

    output = output.detach().cpu().numpy()[0]

    output = resize_to_base( output, 1, base=1, size_out=sz)

    return output


def res_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d( in_channels, in_channels+out_channels, 3, padding=1, stride=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d( in_channels+out_channels, out_channels, 3, padding=1, stride=2),
        nn.LeakyReLU(inplace=True)
    )   

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d( in_channels, in_channels + out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d( in_channels + out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    ) 

def lims_normalize(x):

    #chanMin,_ = x.reshape((*x.shape[0:2], -1)).min(dim=2)
    #x = x - chanMin[:,:,None,None]
    #chanMax,_ = x.reshape((*x.shape[0:2], -1)).max(dim=2)
    #chanMax = torch.clamp(chanMax, 1, 1000)
    #x = x / ( chanMax[:,:,None,None] + 0.001) 
    return x

def resize_to_base(im_in, reduction, base=8, size_out=None):

    if size_out is None:
        sz_in  = np.array(im_in.shape[1:3])[::-1]
        sz_out = sz_in * reduction 
        sz_out = (np.round(sz_out/base)*base).astype(np.int)
    else:
        sz_out = size_out[::-1].astype(np.int)
    
    im_out = np.stack( [cv2.resize(1.*im, tuple(sz_out)) for im in im_in ])
    return im_out

def save_model(model, name, path = ".\\models\\"):

    if not os.path.isdir(path):    os.mkdir(path)
    fn = path + name + ".pt"
    torch.save(model, fn )
