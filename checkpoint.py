# EECS 545 Fall 2020
import itertools
import os
import torch


def save_checkpoint(model, opt, scheduler, epoch, checkpoint_dir, stats):
    """
    Save model checkpoint.
    """
    if scheduler:
        sched_state_dict = scheduler.state_dict()
    else:
        sched_state_dict = None
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'stats': stats,
        'opt_state_dict' : opt.state_dict(),
        'sched_state_dict' : sched_state_dict,
        'module_inits' : dict()
    }

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, opt, scheduler, checkpoint_dir, cuda=True, force=False, pretrain=False, epoch=None):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model, the current epoch, and training losses.
    """
    def get_epoch(cp):
        return int(cp.split('epoch=')[-1].split('.checkpoint.pth.tar')[0])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
    cp_files.sort(key=lambda x: get_epoch(x))

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception('Checkpoint not found')
        else:
            return model, opt, scheduler, 0, []

    # Find latest epoch
    epochs = [get_epoch(cp) for cp in cp_files]
    if epoch is not None:
        inp_epoch = epoch
    else:
        if not force:
            epochs = [0] + epochs
            print('Which epoch to load from? Choose from epochs below:')
            print(epochs)
            print('Enter 0 to train from scratch.')
            print(">> ", end='')
            inp_epoch = int(input())
            if inp_epoch not in epochs:
                raise Exception("Invalid epoch number")
            if inp_epoch == 0:
                print("Checkpoint not loaded")
                clear_checkpoint(checkpoint_dir)
                return model, opt, scheduler, 0, []
        else:
            print('Which epoch to load from? Choose from epochs below:')
            print(epochs)
            print(">> ", end='')
            inp_epoch = int(input())
            if inp_epoch not in epochs:
                raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        stats = checkpoint['stats']
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)".format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise
    
    if opt is not None and checkpoint['opt_state_dict'] is not None:
        opt.load_state_dict(checkpoint['opt_state_dict'])
    if scheduler is not None and checkpoint['sched_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['sched_state_dict'])

    return model, opt, scheduler, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """
    Delete all checkpoints in directory.
    """
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")
