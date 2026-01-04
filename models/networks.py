import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from util.util import to_device, load_net

###############################################################################
# Helper Functions
###############################################################################


def init_wts(net, init_t='normal', gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_t (str)    -- init method: normal | xavier | kaiming | orthogonal
        gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        cls_name = m.__class__.__name__
        if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.Embedding)):
            if init_t == 'N02':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_t in ['glorot', 'xavier']:
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_t == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_t == 'ortho':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('init method [%s] not implemented' % init_t)
    if init_t in ['N02', 'glorot', 'xavier', 'kaiming', 'ortho']:
        net.apply(init_func)
    else:
        net = load_net(net, init_t, 'latest')
    return net

def init_n(net, init_t='normal', gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_t (str)       -- init method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_wts(net, init_t, gain=gain)
    return net


def get_sched(optm, opt):
    """Return a learning rate scheduler

    Parameters:
        optm              -- the optimizer of the network
        opt (option)      -- stores all the experiment flags

    lr_policy: linear | step | plateau | cosine
    """
    if opt.lr_policy == 'linear':
        def lr_lambda(epoch):
            lr = 1.0 - max(0, epoch + opt.epoch_cnt - opt.niter) / float(opt.niter_decay + 1)
            return lr
        sched = lr_scheduler.LambdaLR(optm, lr_lambda=lr_lambda)
    elif opt.lr_policy == 'step':
        sched = lr_scheduler.StepLR(optm, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        sched = lr_scheduler.ReduceLROnPlateau(optm, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        sched = lr_scheduler.CosineAnnealingLR(optm, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('lr policy [%s] not implemented', opt.lr_policy)
    return sched