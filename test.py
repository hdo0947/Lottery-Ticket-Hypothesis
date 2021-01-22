import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

import torch.nn.utils.prune as prune

from train import train, train_kd
from models import ffn
from prune_fns import prune_model

from torchvision import transforms

from dataset import MNIST_Dataset


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
config = {
    'num_epoch' : 1,
    'lr' : 0.001,
    'beta1' : 0.9,
    'batch_size' : 64,
    'ckpt_path' : 'checkpoints/pruning_test/fc_0/',
    'alpha' : 0.75,
    'val_freq' : 640,
    'device' : device
}
dataset = MNIST_Dataset(config['batch_size'], device=config['device'], val=0.1)

model = ffn(784, 64, 10, device).to(device)

model0, model_loss0, model_acc0, iterations, test_loss, test_acc, train_loss = train(config, dataset, model)

p=0.2
model_copy0 = ffn(784, 16, 10, device).to(device)
model_copy0.copy_weights(model)
prune_model(model, 0.2)
model0, model_loss0, model_acc0, iterations, test_loss, test_acc, train_loss = train(config, dataset, model)

config['ckpt_path'] = 'checkpoints/pruning_test/fc_1_p20/'
model1, model_loss1, model_acc1, iterations = train_kd(config, dataset, model, model_copy0)

prune_model(model, p)
config['alpha'] = 1.0
model2, model_loss2, model_acc2, iterations, test_loss, test_acc = train_kd(config, dataset, model, model_copy0)
plt.plot(iterations, model_acc0, iterations, model_acc1, iterations, model_acc2)
plt.show()