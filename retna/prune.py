import numpy as np 
import torch.nn as nn

import torch 


def reduce_model(model):

    dependacy = make_dep_list(model)

    Model_Parameters =  [param for param in model.parameters()]
    weights = Model_Parameters[::2]
    bias =  Model_Parameters[1::2]

    unused,dep_weights = get_unused_inputs(weights)
    out_mask, in_mask = get_weight_masks(weights, dependacy, unused)
    weights_out, bias_out = prune_weights(weights, bias, in_mask, out_mask)

    model = write_weights_to_model(model, weights_out, bias_out )

    return model,dep_weights


def write_weights_to_model(model,weights,bias ):
    n = 0 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight = nn.Parameter(weights[n].data.clone())
            if m.bias is not None:
                m.bias = nn.Parameter(bias[n].data.clone())
                n +=1
    return model 

## For vnet only 
def make_dep_list(model):
    n_modules = int(len(list(model.parameters()))/2)

    dependacy = [[] for _ in range(n_modules) ]
    for i in range(n_modules):

        dependacy[i].append(i-1)

        if i%2 == 0 and i != (n_modules-1):
            
            dependacy[-1].append(i-1)
    return dependacy

def make_dependacy_list(up_cat_list,weights):
    
    dependacy_list = [[i-1] if i>0 else [] for i in range(len(weights)) ]
    if len(up_cat_list)>0:
        [dependacy_list[i[0]].extend([i[1]]) for i in up_cat_list][0]

    compute_edges_forw = [[i+1] if i<(len(weights)-1) else [] for i in range(len(weights)) ]
    if len(up_cat_list)>0:
        [compute_edges_forw[i[1]].extend([i[0]]) for i in up_cat_list][0]
    
    return dependacy_list

def get_unused_inputs(weights):

    unused = [ np.zeros(w.shape[1],dtype=bool) for w in weights ]

    in_weights = []
    for i in range(len(weights)):
        i = -(i+1)
        W = weights[i].detach().numpy()   
        
        W2 = np.abs(W).max(axis=(2,3))
        W_max = W2.max(axis=0)
        W_max = W_max/W_max.max()
        
        drop_idx = W_max < 0.5
        unused[i]= drop_idx
        in_weights.append(W_max)

    return unused, in_weights

def get_weight_masks(weights, dependacies, unused):

    out_mask = [ np.ones(w.shape[0],dtype=bool) for w in weights ]
    out_mask[-1] = ~out_mask[-1][:]
    for i in range(1,len(weights)):
        i = -(i+1)
        
        s_idx = 0 
        for idx in dependacies[i]:
            if idx>=0:
                end_idx = s_idx+len(weights[idx])
                out_mask[idx] = out_mask[idx] & (unused[i][s_idx:end_idx])
                s_idx = end_idx   

    out = []
    for mask in out_mask:  
        if all(mask): mask[0]=False
        out.append(mask)
    out_mask = out    
    
    in_mask = out_mask_to_in_mask(weights, dependacies, out_mask)

    return out_mask, in_mask

def out_mask_to_in_mask(weights, dependacies, out_mask):

    in_mask = [ np.zeros(w.shape[1],dtype=bool) for w in weights ]
    src_mask = np.zeros(weights[0].shape[1],dtype=bool)
    for i in range(1,len(weights)):
        mask = [ out_mask[idx] if idx>=0 else src_mask for idx in dependacies[i] ]
        in_mask[i] = np.concatenate(mask)

    return in_mask

def prune_weights(weights, bias, in_mask, out_mask):

    weights_out = [[] for i in range(len(weights)) ]
    bias_out = [[] for i in range(len(weights)) ] 

    for i, mask in enumerate(out_mask):
        weights_out[i] = weights[i][mask==False,:,:,:]    
        bias_out[i] = bias[i][mask==False]
        
    for i, mask in enumerate(in_mask):
        weights_out[i] = weights_out[i][:,mask==False,:,:]     
        
    return weights_out, bias_out


## 

def mask_layers(weights, bias, input_mask, out_mask):

    weights_out = weights
    bias_out = bias

    for i, mask in enumerate(out_mask):
        
        req_grad = weights_out[i].requires_grad
        if req_grad:   weights_out[i].requires_grad = False
        weights_out[i][mask,:,:,:] = 0
        weights_out[i][:,input_mask[i],:,:] = 0     
        if req_grad:   weights_out[i].requires_grad = True
        
        ##
        req_grad = bias_out[i].requires_grad
        if req_grad:   bias_out[i].requires_grad = False
        bias_out[i][mask] = 0       
        if req_grad:   bias_out[i].requires_grad = True
            
    return weights_out, bias_out

def grow_layers(weights,bias, dependacy_list, grow_list=None):


    if grow_list == None:
        grow_list = [True for _ in range(len(weights))]
        grow_list[-1] = False 


    weights_out = [[] for i in range(len(weights)) ]
    bias_out = [[] for i in range(len(weights)) ] 

    for i, grow in enumerate(grow_list):
        
        w = weights[i]
        b = bias[i]
        if grow:

            new_weights = torch.rand(2,w.shape[1],w.shape[2],w.shape[3])*0.1
            new_biases = torch.rand(2)*0.1

            w = torch.cat(( w,  new_weights ), dim=0 )
            b = torch.cat(( b,  new_biases ) , dim=0 ) 
            
        if dependacy_list[i]:
                       
            w_in = w
            w = []
            s_idx = 0
            for idx in dependacy_list[i]:
                
                end_idx = s_idx +  weights[idx].shape[0]
                w.append( w_in[:, s_idx:end_idx,:,:] )
                s_idx = end_idx   
                print(idx)   
                if idx>=0 and grow_list[idx]:
                    new_weights = torch.rand( w_in.shape[0], 2 , w_in.shape[2], w_in.shape[3]) * 0.1
                    w.append( new_weights )

            w = torch.cat(w,dim=1)
        weights_out[i] = w
        bias_out[i] = b

    return weights_out, bias_out


def add_block(model, HL_1=8, HL_2 = 8):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    blocks = model.blocks
    block = res_conv( HL_1, HL_2)
    blocks.append( block )

    w = model.conv_last.weight
    new_nuerons = torch.rand( w.shape[0], HL_1 , w.shape[2], w.shape[3])*0.1
    new_nuerons = new_nuerons.to(device)
    x = torch.cat(( w, new_nuerons ), dim=1 )
    model.conv_last.weight = nn.Parameter(x.data.clone())
    model = model.to(device)
    
    return model

def add_output_channel(model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    w = model.conv_last.weight
    b = model.conv_last.bias

    w2 = torch.cat(( w, w ), dim=0 )
    b2 = torch.cat(( b, b ), dim=0 )
    
    model.conv_last.weight = nn.Parameter(w2.data.clone())
    model.conv_last.bias   = nn.Parameter(b2.data.clone())
    model = model.to(device)
        
    return model


def select_output_channel(model, idx):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    w = model.conv_last.weight
    b = model.conv_last.bias

    w = w[np.array(idx)]
    b = b[np.array(idx)]
    
    model.conv_last.weight = nn.Parameter(w.data.clone())
    model.conv_last.bias   = nn.Parameter(b.data.clone())
    model = model.to(device)
        
    return model