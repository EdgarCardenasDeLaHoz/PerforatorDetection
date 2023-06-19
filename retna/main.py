from .networks import *
from .training import *
from .data_loader import *
from .view import *
from .prune import *
from .hook_tools import *
from .simulate_data import *

import torch
import glob as glob
import matplotlib.pyplot as plt
import matplotlib as mpl

class Main():
    def __init__(self, path_data, path_model, chans, size=[100,100], label=False, noise=False, batch=1,rnd_noise=0.3, noise_range=[.1,.5],ns_par=[0,0.3,1]):
        self.path_data = path_data
        self.path_model = path_model
        self.h_chans = chans
        self.force_noise = noise
        self.batch = batch
        self.size = size
        self.label = label
        self.rand_noise = rnd_noise
        self.range_noise = noise_range
        self.ns_param = ns_par
    
    def training(self, num, load=False, loss = 10, train = 20, cycles = 20):
        if load == True:
            print("getting model")
            self.call_trained(self.path_model)
        else:
            self.model = Retna_V1(3,6, self.h_chans)
            self.optimizer = None
            self.epoch = 0
        self.loader()

        self.model = self.model.to("cuda:0")
        self.Loader.dataset.device = "cuda:0"
        self.model, self.optimizer, self.epoch = train_model(self.model, self.Loader, optimizer=self.optimizer, cur_epoch=self.epoch, num_epochs=num, n_select_loss=loss, n_select_train=train, n_train_cycles=cycles)
        self.model.save("train")
        checkpoint = {
            'model': self.model, 
            'optimizer': self.optimizer,
            'epoch': self.epoch
            }
        torch.save(checkpoint,'./models/checkpoint.pt')

    def call_trained(self, path):
        info = torch.load(path)
        self.model = info['model']
        self.optimizer = info['optimizer']
        self.epoch = info['epoch']
    
    def loader(self):
        self.datas =  glob.glob(self.path_data)
        self.datas = Image_Handler(self.datas)
        self.Loader = DataLoader(self.datas, batch_size=self.batch)
        #####################parameters#####################
        self.Loader.dataset.outsize = self.size
        self.Loader.dataset.force_label = self.label
        self.Loader.dataset.noise = self.force_noise
        self.Loader.dataset.nois_low = self.ns_param[0]
        self.Loader.dataset.noise_high = self.ns_param[1]
        self.Loader.dataset.noise_size = self.ns_param[2]
        self.Loader.dataset.scale_range = [.2,.8]
        self.Loader.dataset.noise_range = self.range_noise

    def mosaic(self):
        print_mosaic(self.Loader,self.model)
        plt.show()

    def cam_predict(self, crop, path):
        self.call_trained(path)
        self.Loader = DataLoader
        self.Loader.dataset.randomize = False
        self.Loader.dataset.mosaic = True
        self.Loader.dataset.random_selection = False
        self.Loader.dataset.in_memory = []
        self.Loader.dataset.selection_size = 1

        self.Loader.dataset.device="cpu"

        if model is not None:  model = model.to("cpu")

        pred = model(crop)
        P = colorize_channels(pred)
        return P

if __name__ == "__main__":
    data_path = r"C:\Users\mheva\OneDrive\Bureaublad\temp/*"
    model_path = r"C:\Users\mheva\OneDrive\Documents\GitHub\Retna\models\checkpoint.pt"
    call = Main(data_path, model_path, [50,30,30,20,20,40], size=[150,150], label=True, batch=2, rnd_noise=0.3, noise=True, ns_par=[0,0.3,1])
    #for i in range(10):
    #    try: call.training(50, True, 10, 100, 10)
    #    except: call.training(50, False, 10, 100, 10)
    call.call_trained(model_path)
    call.mosaic()


#desirable idea, creating a GUI that you are easily able to switch between:
# model type
# does the database need to have a label in it yes or no?
# how many cycles?
# how many train selects?
# how many loss selects?
# how many epochs
# what batch size?
# how many hidden channels
# what is the base size of the data loaded (you can decide it yourself within the space of the picture base recommend is 150x150)
# explorer window to find the folder with the database
# explorer window to find the pt file IF IT EXISTS IN THE FIRST PLACE!!! MAKE A CHECK FOR REDUNDANCY!!
# a select window with all sorts of options of manipulations like:
# skew - more difficult then it was expected to apply
# scale
# noise
# many others
# give a information window or make it so that a button opens a txt file with all the necessary information about the functions and or
# general information about the structure of the AI code.