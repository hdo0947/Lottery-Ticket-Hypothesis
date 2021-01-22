import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model, p, params_to_prune_fn, device, layer_wise=False, start_at=0, method=prune.L1Unstructured):
    """
    prune the lowest p% of weights by magnitude
    """
    # specify the layers to consider for this pruning:

    
    params_to_prune = params_to_prune_fn(model)
    # finds the masks for the above modules (i.e. layers) and saves 
    # them as attributes with name `module_name`+'_mask', where 
    # `module_name` is the 2nd arg in each element of parameters_to_prune.
    
    prune.global_unstructured(params_to_prune, pruning_method=method, amount=p)
    
    # apply the masks to name_init attributes, and update name with
    # the new masked weights:
    with torch.no_grad():
        for module, name in params_to_prune:
            
            # check for the weight tensor in model.rewind_state_dict:
            mask = getattr(module, name + "_mask")
            orig = getattr(module, name + "_init").to(device)
            pruned_tensor = mask.to(dtype=orig.dtype) * orig
            
            # replace masked weight with masked initial weight:
            setattr(module, name, pruned_tensor)
            setattr(module, name+"_orig", torch.nn.Parameter(orig))

def global_prune(model, p, params_to_prune_fn):
    # Equivalent to prune.global_unstructured, but doesn't store masks.
    # Overall, this is less useful than prune.global_unstructured.
    
    weight_tensor = torch.nn.utils.parameters_to_vector(getattr(module[0], module[1]) for module in params_to_prune_fn(model))
    
    amount_prune = int(weight_tensor.nelement() * p)
    
    weight_mags = torch.abs(weight_tensor).view(-1)
    weights_to_prune = torch.topk(weight_mags, k=amount_prune, largest=False)
    
    mask = torch.ones_like(weight_tensor)
    mask.view(-1)[weights_to_prune.indices] = 0
    
    # reform individual modules from weight_tensor:
    
    counter = 0
    for module, name in params_to_prune_fn(model):
        weight = getattr(module, name)
        
        weight_len = param.numel()
        
        # slice weight_tensor:
        mask_slice = mask[counter : counter + weight_len].view_as(weight)
        
        weight_init = getattr(module, name+"_init")
        new_weight = mask_slice * weight
        # replace module.name.weight with the pruned tensor
        setattr(module, name, torch.nn.Parameter(new_weight))
        counter += weight_len