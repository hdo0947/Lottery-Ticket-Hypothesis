import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ffn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
        # (optional) initialize weights
        # here, just using default pytorch init
        
        # save initial weights for each layer as attribute:
        #checkpoint_weights(model, "init")
        
    def copy_weights(self, other):
        # copy weight and bias attributes for each named module in other to self
        # should make sure other is same type as self
        first_iter=True
        for module in self.named_modules():
            if first_iter:
                first_iter=False
                continue
                
            other_weight = other._modules[module[0]].weight.detach().clone()
            other_bias = other._modules[module[0]].bias.detach().clone()
            setattr(module[1], "weight", torch.nn.Parameter(other_weight))
            setattr(module[1], "bias", torch.nn.Parameter(other_bias))
            setattr(module[1], "weight_init", other._modules[module[0]].weight_init.detach().clone())
            setattr(module[1], "bias_init", other._modules[module[0]].bias_init.detach().clone())
        return
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = F.relu(self.dropout(self.fc1(x)))
        z = F.relu(self.dropout(self.fc2(z)))
        z = F.relu(self.dropout(self.fc3(z)))
        return self.dropout(self.fc4(z))

def checkpoint_weights(model, params_fn, subscript="_init"):
    """
    Saves all parameters in model, where layer's have parameters "weight" and "bias",
    to names "weight_<subscript>", "bias_<subscript>"
    """
    params = params_fn(model)
    first_iter = True
    with torch.no_grad():
        for module in params:
            # otherwise, set an attribute for the initial weights:
            if module[1] == "weight":
                setattr(module[0], "weight"+subscript, module[0].weight.detach().clone())
            elif module[1] == "bias":
                setattr(module[0], "bias"+subscript, module[0].bias.detach().clone())
            
def load_resnet18(num_out_class, device, pretrained=False):
    resnet18 = torchvision.models.resnet18(pretrained=pretrained, progress=True)
    resnet18.fc = nn.Linear(512, num_out_class)
    resnet18 = resnet18.to(device)
    return resnet18
    