
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def add_hooks(model):
    forw_hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            forw_hooks.append(Hook(m))
        model.hooks = forw_hooks

def close_hooks(model):

    f_hook = model.hooks
    [hk.close() for hk in f_hook]
    model.hooks = []

def view_forward_layer(model,Loader,n):

    add_hooks(model)

    Loader.dataset.randomize = False
    Loader.dataset.random_selection = True
    Loader.dataset.selection_size = 1
    Loader.dataset.in_memory = []
    model.eval()

    f_hook = model.hooks
    for inputs,_,_ in Loader:
        pred = model(inputs) 
        for hook in [f_hook[n]]: 
            draw_hook(hook)
        Loader.dataset.in_memory = []

        
    close_hooks(model)
        
def get_output_map(model, Loader, n=20, draw=True):

    Loader.dataset.randomize = False
    Loader.dataset.random_selection = True
    Loader.dataset.selection_size = n
    model.eval()

    add_hooks(model)

    f_hook = model.hooks    
    outputs = [[] for i in range(len(f_hook))]

    for inputs,_,_ in Loader:
        Loader.dataset.in_memory = []
        _ = model(inputs)
        for i, hook in enumerate(f_hook):

            im = np.array(hook.output[0].detach())
            im = np.abs(im)
            res = im.max(axis=(1,2))

            if len(outputs[i])==0:
                outputs[i] = res
            else: 
                outputs[i] += res 
    
    close_hooks(model)
    if draw:
        draw_output_map(outputs)   

    return outputs

def draw_output_map(outputs):
    
    fig = plt.figure() 
    m = len(outputs)
    for i in range(len(outputs)):
        ax = fig.add_subplot(1,m,i+1) 
        plt.title(str(i))
        ax.imshow(np.array([outputs[i]]).T)

def draw_hook(hook):

    hk = hook.output[0].detach().cpu().numpy()
    im = hk.transpose([1,2,0])

    fig = plt.figure()
    #plt.tight_layout()
    fig.subplots_adjust(left=.05, bottom=.05, 
                right=.95, top=.95, 
                wspace=.1, hspace=.1)
    num_layers = im.shape[2]
    for i in range(num_layers):

        m = np.ceil(np.sqrt(num_layers))
        n = np.round(np.sqrt(num_layers))
        ax = fig.add_subplot(m,n,i+1)

        I = im[:,:,i]
        ax.imshow(I)

def make_summary_plots(all_df):
    summary_plot(all_df)
    plt.savefig(root+"All_images.jpg")

    names = all_df["filename"]
    marker = [fn.split("_")[1] for fn in names]
    marker, m_idx = np.unique(marker,return_inverse=True)
    for n,mark in enumerate(marker):
        df = all_df[m_idx==n]
        summary_plot(df, mark)
        plt.savefig(root+ mark +".jpg")

def summary_plot(all_df, title=None):
    
    names = all_df["filename"]
    cond = [fn.split("_")[0] for fn in names]
    rep  = [fn.split("_")[2] for fn in names]
    uni_cond, x_idx = np.unique(cond,return_inverse=True)
    u_rep,r_idx  = np.unique(rep,return_inverse=True)
    
    columns = ["Nuclei_mean_GFP",	"Nuclei_mean_TXRED",	
               "Cytoplasm_mean_GFP",	"Cytoplasm_mean_TXRED"] 
    
    fig,axs = plt.subplots(2,2,figsize=(36,18))
    plt.subplots_adjust(  left=0.04, bottom=0.06, right=0.98, top=0.96, 
                              wspace = 0.08, hspace = 0.22 ) 

    if title is not None:
        fig.suptitle(title)
    
    for n,col in enumerate(columns):

        ax = axs.ravel()[n]
        Y = all_df[col]
        for i in range(len(u_rep)):
            x = x_idx[r_idx==i]
            y = Y[r_idx==i]
            ax.scatter(x, y, label=u_rep[i], s=70, edgecolors='none')
        
        ax.set_title(col, fontsize=20)
        ax.set_xticks(np.arange(0,len(uni_cond)+1))
        ax.set_xticklabels(uni_cond, rotation=30, fontsize=20)
        ax.legend()
        ax.grid(True)
        
    
       
