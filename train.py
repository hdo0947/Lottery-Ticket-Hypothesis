import torch
import torch.nn as nn
import torch.nn.functional as F
import checkpoint
import matplotlib.pyplot as plt

from models import checkpoint_weights

def predictions(logits):
    return torch.argmax(logits, dim=1)

def _running_accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().item()

def loss_kd(y_pred, y_true, y_teacher, crit_ground_truth, crit_teacher, alpha):
    return alpha*crit_ground_truth(y_pred, y_true) + (1-alpha)*crit_teacher(y_pred, y_teacher)

def softCEL(pred, target):
    # finds the cross entropy loss for pred, target, where target is a distribution
    # i.e. soft labels
    log_probs_pred = F.log_softmax(pred, dim=1)
    probs_target = F.softmax(target, dim=1)
    return -1*torch.sum(log_probs_pred * probs_target) / pred.shape[0]

def test(model, test_loader, crit, device):
    model.eval()
    acc = 0
    loss = 0
    num_points = 0
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(X)
            pred = predictions(output.data) 
            loss += crit(output, y).item() * X.size(0)
            acc += _running_accuracy(pred, y)
            num_points += y.shape[0]
    acc = acc / num_points
    model.train()
    return acc, loss


def avg(inp):
    return sum(inp) / len(inp)

def train(config, dataset, model, save_weights_iter=None, params_to_save=None, save_progress=True):
    
    lr = config['lr']
    beta1 = config['beta1']
    device = config['device']
    val_freq = config['val_freq']
    batch_size = config['batch_size']
    opt = config['opt']
    scheduler = config['scheduler']
    
    crit = config['crit']
    
    train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    
    if save_progress:
        model, opt, scheduler, start_epoch, stats = checkpoint.restore_checkpoint(model, opt, scheduler, config['ckpt_path'], force=force, pretrain=False)
    else:
        start_epoch=0
    
    if not stats:
        model_loss = []
        model_acc = []
        iterations = []
    else:
        model_loss, model_acc, iterations = stats
    itr = 0

    train_loss = []

    for epoch in range(start_epoch, config['num_epoch']):
        model.train()
        epoch_train_loss = []
        for i, (X, y) in enumerate(train_loader):
            if (save_weights_iter or save_weights_iter==0) and itr == save_weights_iter:
                checkpoint_weights(model, params_to_save)
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            output = model(X).to(device)
            loss = crit(output, y)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            epoch_train_loss.append(loss.item())
            acc, loss = 0, 0
            if itr % val_freq == 0:
                acc, loss = test(model, val_loader, crit, device)
                model_acc.append(acc)
                model_loss.append(loss)
                iterations.append(itr)

                print(itr, epoch, acc, loss, avg(epoch_train_loss))          

            itr += 1
            
        stats = [model_loss, model_acc, iterations]
        if save_progress:
            checkpoint.save_checkpoint(model, opt, scheduler, epoch + 1, config['ckpt_path'], stats)
        if scheduler:
            scheduler.step()
    
    test_acc, test_loss = test(model, test_loader, crit, device)
    return [model, model_loss, model_acc, iterations, test_loss, test_acc, train_loss]

def train_kd(config, dataset, model, save_weights_iter=None, params_to_save=None, save_progress=True):
    lr = config['lr']
    beta1 = config['beta1']
    device = config['device']
    val_freq = config['val_freq']
    batch_size = config['batch_size']
    opt = config['opt']
    scheduler = config['scheduler']
    
    crit = config['crit']
    crit_teacher = config['crit_teacher']
    
    train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    
    if save_progress:
        model, opt, scheduler, start_epoch, stats = checkpoint.restore_checkpoint(model, opt, scheduler, config['ckpt_path'], force=force)
    else:
        start_epoch=0
    
    if not stats:
        model_loss = []
        model_acc = []
        iterations = []
    else:
        model_loss, model_acc, iterations = stats
    itr = 0

    train_loss = []

    for epoch in range(start_epoch, config['num_epoch']):
        model.train()
        epoch_train_loss = []
        for i, (X, y) in enumerate(train_loader):
            if (save_weights_iter or save_weights_iter==0) and itr == save_weights_iter:
                checkpoint_weights(model, params_to_save)
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            output = model(X).to(device)
            loss = crit_teacher(X, output, y)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            epoch_train_loss.append(loss.item())
            acc, loss = 0, 0
            if itr % val_freq == 0:
                acc, loss = test(model, val_loader, crit, device)
                model_acc.append(acc)
                model_loss.append(loss)
                iterations.append(itr)

                print(itr, epoch, acc, loss, avg(epoch_train_loss))

            itr += 1
            
        stats = [model_loss, model_acc, iterations]
        if save_progress:
            checkpoint.save_checkpoint(model, opt, scheduler, epoch + 1, config['ckpt_path'], stats)
        if scheduler:
            scheduler.step()
    
    test_acc, test_loss = test(model, test_loader, crit, device)
    return [model, model_loss, model_acc, iterations, test_loss, test_acc, train_loss]

class KnowledgeDistill_Loss(nn.Module):
    def __init__(self, teacher, alpha):
        super(KnowledgeDistill_Loss, self).__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, X, y_pred, y_true):
        y_teacher = self.teacher(X.detach())
        
        loss = self.loss(y_pred, y_true)
        
        log_probs_pred = F.log_softmax(y_pred, dim=1)
        probs_target = F.softmax(y_teacher, dim=1)
        kd_loss = -1*torch.sum(log_probs_pred * probs_target) / y_pred.shape[0] 
        
        return torch.add(self.alpha*loss, (1-self.alpha)*kd_loss)
