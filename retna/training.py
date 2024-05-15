
import numpy as np 
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from .loss import calc_loss

def train_model(model, Loader, optimizer=None, cur_epoch = 0, num_epochs=10, n_select_loss = 10, n_select_train = 20, n_train_cycles = 20):

    model.to( Loader.dataset.device )

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_model_wts = copy.deepcopy(model.state_dict())

    Loader.dataset.reset_weights()
    best_loss = evaluate_model(model,Loader, n_select_loss)
    improved_count = [0,0]
    
    print("Start Loss:" , best_loss)
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(cur_epoch+epoch+1, cur_epoch+ num_epochs), end = "\t")

        model = iterate_epoch(model,Loader,optimizer, n_select_train, n_train_cycles)
        #evaluate_model(model,Loader, n_select=10)

        losses = Loader.dataset.selection_weights
        epoch_loss = losses[losses<1].mean()

        print("Loss: {:4f}".format(epoch_loss), end = "\t")
        
        b_improved = epoch_loss < best_loss
        if b_improved:
            print("Caching improved model", end = "\t")
            best_model_wts = copy.deepcopy(model.state_dict()) 
            best_loss = epoch_loss

        if np.mod(epoch,10)==0 and (epoch>10):
            bRand = np.random.rand(*losses.shape) < 0.5
            losses[bRand] = 1
            
        #losses[np.isnan(losses)] = 1

        Loader.dataset.selection_weights = losses

        optimizer, improved_count = adjust_optimizer( optimizer, improved_count, b_improved)
        print("")
    
    cur_epoch += num_epochs
        
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, optimizer, cur_epoch

def iterate_epoch(model, Loader , optimizer, n_select=10, n_cycles=10):

    Loader.dataset.randomize = True
    Loader.dataset.random_selection = True
    Loader.dataset.selection_size = n_select
    method = Loader.dataset.method
    sel_chan = Loader.dataset.train_channels
    

    ###
    model.train()
    n_samples, total_loss = 0,0
    for n in range(n_cycles):
        Loader.dataset.in_memory = []
        print(".", end="")
        for inputs,labels,idx in Loader:
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)
                loss = calc_loss(outputs, labels, method, sel_chan)
                loss.backward()
                optimizer.step()

            np_loss = loss.data.cpu().numpy()

            if n==0:
                Loader.dataset.selection_weights[idx] = np_loss

            losses = np_loss * labels.size(0)
            total_loss += losses
            n_samples += inputs.size(0)

    #loss = total_loss / n_samples

    return model

def evaluate_model(model, Loader, n_select=10):

    Loader.dataset.randomize = False
    Loader.dataset.random_selection = False
    Loader.dataset.mosaic = True
    Loader.dataset.selection_size = n_select
    method = Loader.dataset.method
    sel_chan = Loader.dataset.train_channels

    model.eval()

    #######
    n_samples, total_loss = 0,0
    Loader.dataset.in_memory = []

    for inputs,labels,idx in Loader:
        #print(np.array(idx),end="")

        with torch.no_grad():
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, method, sel_chan)

        Loader.dataset.in_memory = []

        np_loss = loss.data.cpu().numpy()
        Loader.dataset.selection_weights[idx] = np_loss

        total_loss += (np_loss * labels.size(0))
        n_samples += inputs.size(0)
        

    losses = Loader.dataset.selection_weights
    return losses[losses<1].mean()

def adjust_optimizer( optimizer, improved_count, b_improved ):

    if b_improved:
        improved_count[0] += 1
        improved_count[1] = 0
    else:
        improved_count[0] = 0
        improved_count[1] += 1

    if  improved_count[1] %3==0 and improved_count[1] >0:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.8
        print("lr:", optimizer.param_groups[0]["lr"]  , end = "\t")

    if  optimizer.param_groups[0]["lr"] < 5e-5:
        optimizer.param_groups[0]["lr"] = 1e-3
        print("lr:", optimizer.param_groups[0]["lr"] , end = "\t")

    return optimizer, improved_count

############################################################

def train_model_x(model, Loader, optimizer=None, num_epochs=10):

    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = evaluate_x(model,Loader, n_select=10)
    improved_count = [0,0]

    print("Start Loss:" , best_loss)
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1), end = "\t")

        model = iterate_x(model,Loader,optimizer, n_select=10, n_cycles=10)
        epoch_loss = evaluate_x(model,Loader, n_select=10)


        print("Loss: {:4f}".format(epoch_loss), end = "\t")
        b_improved = epoch_loss < best_loss
        if b_improved:
            print("Caching improved model", end = "\t")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict()) 

        optimizer, improved_count = adjust_optimizer( optimizer, improved_count, b_improved)
        print("")
        
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def iterate_x(model, Loader , optimizer, n_select=10, n_cycles=10):
    
    ###
    model.train()
    n_samples, total_loss = 0,0
    for n in range(n_cycles):
        
        print(".", end="")
        for X in Loader:
            if len(X)==3:   inpt,targ,idx = X
            elif len(X)==2: inpt,targ = X

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inpt)
                outputs = torch.Tensor(outputs)
                loss = calc_loss(outputs, targ)
                loss.backward()
                optimizer.step()

            np_loss = loss.data.cpu().numpy()

            losses = np_loss * targ.size(0)
            total_loss += losses
            n_samples += targ.size(0)

    #loss = total_loss / n_samples

    return model

def evaluate_x(model, Loader, n_select=10):

    model.eval()

    #######
    n_samples, total_loss = 0,0
    
    for X in Loader:
        if len(X)==3:   inpt,targ,idx = X
        elif len(X)==2: inpt,targ = X
        
        #print(np.array(idx),end="")

        with torch.no_grad():
            outputs = model(inpt)
            loss = calc_loss(outputs, targ)
            
        np_loss = loss.data.cpu().numpy()
        losses = np_loss * targ.size(0)
        total_loss += losses
        n_samples += inpt.size(0)

        
    loss = total_loss / n_samples

    return loss