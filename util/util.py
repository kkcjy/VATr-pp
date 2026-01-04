"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def load_net(net, save_dir, epoch):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    load_file = '%s_net_%s.pth' % (epoch, net.name)
    load_path = os.path.join(save_dir, load_file)
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state = torch.load(load_path)
    if hasattr(state, '_metadata'):
        del state._metadata
    net.load_state_dict(state)
    return net

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)

def multi_replace(s, rep_dict):
    for key in rep_dict.keys():
        s = s.replace(key, rep_dict[key])
    return s

def get_batch(data, bs, cnt):
    batch = {}
    for key in data:
        batch[key] = data[key][bs*cnt:bs*(cnt+1)]
    return batch

# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# turn tensor of classes to tensor of one hot tensors:
def make_one_hot(labels, lens, n_classes):
    one_hot = torch.zeros((labels.shape[0], labels.shape[1], n_classes),dtype=torch.float32)
    for i in range(len(labels)):
        one_hot[i,np.array(range(lens[i])), labels[i,:lens[i]]-1]=1
    return one_hot

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, len_fake, lens, mask):
    try:
        mask_real = torch.ones(dis_real.shape).to(dis_real.device)
        mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    except RuntimeError:
        raise
    if mask and len(dis_fake.shape)>2:
        for i in range(len(lens)):
            mask_real[i, :, :, lens[i]:] = 0
            mask_fake[i, :, :, len_fake[i]:] = 0
    loss_real = torch.sum(F.relu(1. - dis_real * mask_real))/torch.sum(mask_real)
    loss_fake = torch.sum(F.relu(1. + dis_fake * mask_fake))/torch.sum(mask_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake, len_fake, mask):
    mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    if mask and len(dis_fake.shape)>2:
        for i in range(len(len_fake)):
            mask_fake[i, :, :, len_fake[i]:] = 0
    loss = -torch.sum(dis_fake*mask_fake)/torch.sum(mask_fake)
    return loss

def loss_std(z, lens, mask):
    loss_std = torch.zeros(1).to(z.device)
    z_mean = torch.ones((z.shape[0], z.shape[1])).to(z.device)
    for i in range(len(lens)):
        if mask:
            if lens[i]>1:
                loss_std += torch.mean(torch.std(z[i, :, :, :lens[i]], 2))
                z_mean[i,:] = torch.mean(z[i, :, :, :lens[i]], 2).squeeze(1)
            else:
                z_mean[i, :] = z[i, :, :, 0].squeeze(1)
        else:
            loss_std += torch.mean(torch.std(z[i, :, :, :], 2))
            z_mean[i,:] = torch.mean(z[i, :, :, :], 2).squeeze(1)
    loss_std = loss_std/z.shape[0]
    return loss_std, z_mean

# Convenience utility to switch off requires_grad
def toggle_grad(model, on_off):
    for param in model.parameters():
        param.requires_grad = on_off


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
        # Only apply this to parameters with at least 2 axes, and not in the blacklist
        if len(param.shape) < 2 or any([param is item for item in blacklist]):
            continue
        w = param.view(param.shape[0], -1)
        grad = (2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w))
        param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                    - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += strength * grad.view(param.shape)

# A highly simplified convenience class for sampling from distributions
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
class Dist(torch.Tensor):
    # Init the params of the distribution
    def init_dist(self, dist_type, **kwargs):
        seed_rng(kwargs['seed'])
        self.dist_type = dist_type
        self.kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.n_cats = kwargs['num_categories']
        elif self.dist_type == 'poisson':
            self.lam = kwargs['var']
        elif self.dist_type == 'gamma':
            self.scale = kwargs['var']


    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.n_cats)
        elif self.dist_type == 'poisson':
            type = self.type()
            dev = self.device
            data = np.random.poisson(self.lam, self.size())
            self.data = torch.from_numpy(data).type(type).to(dev)
        elif self.dist_type == 'gamma':
            type = self.type()
            dev = self.device
            data = np.random.gamma(shape=1, scale=self.scale, size=self.size())
            self.data = torch.from_numpy(data).type(type).to(dev)

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Dist(self)
        new_obj.init_dist(self.dist_type, **self.kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def to_dev(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids)>1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    return net


# Convenience function to prepare a z and y vector
def prep_z_y(bs, dim_z, n_classes, device='cuda',
                fp16=False, z_var=1.0, z_dist='normal', seed=0):
    z_ = Dist(torch.randn(bs, dim_z, requires_grad=False))
    z_.init_dist(z_dist, mean=0, var=z_var, seed=seed)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Dist(torch.zeros(bs, requires_grad=False))
    y_.init_dist('categorical', num_categories=n_classes, seed=seed)
    y_ = y_.to(device, torch.int64)
    return z_, y_


def tensor2im(img_tensor, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        img_tensor (tensor) --  the input image tensor array
        imtype (type)       --  the desired type of the converted numpy array
    """
    if not isinstance(img_tensor, np.ndarray):
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.data
        else:
            return img_tensor
        img_np = img[0].cpu().float().numpy()
        if img_np.shape[0] == 1:
            img_np = np.tile(img_np, (3, 1, 1))
        img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        img_np = img_tensor
    return img_np.astype(imtype)


def diagnose_net(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    cnt = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            cnt += 1
    if cnt > 0:
        mean = mean / cnt
    print(name)
    print(mean)


def save_img(img_np, img_path):
    """Save a numpy image to the disk

    Parameters:
        img_np (numpy array) -- input numpy array
        img_path (str)       -- the path of the image
    """
    img_pil = Image.fromarray(img_np)
    img_pil.save(img_path)


def print_np(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)