import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PosEnc1D(nn.Module):
    def __init__(self, ch):
        """
        :param ch: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PosEnc1D, self).__init__()
        self.org_ch = ch
        ch = int(np.ceil(ch / 2) * 2)
        self.ch = ch
        inv_freq = 1.0 / (10000 ** (torch.arange(0, ch, 2).float() / ch))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cache = None

    def forward(self, x):
        """
        :param x: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(x.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cache is not None and self.cache.shape == x.shape:
            return self.cache

        self.cache = None
        bs, x_dim, orig_ch = x.shape
        pos_x = torch.arange(x_dim, device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x_dim, self.ch), device=x.device).type(x.type())
        emb[:, : self.ch] = emb_x

        self.cache = emb[None, :, :orig_ch].repeat(bs, 1, 1)
        return self.cache


class PosEncPerm1D(nn.Module):
    def __init__(self, ch):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PosEncPerm1D, self).__init__()
        self.penc = PosEnc1D(ch)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc = self.penc(x)
        return enc.permute(0, 2, 1)

    @property
    def org_ch(self):
        return self.penc.org_ch


class PosEnc2D(nn.Module):
    def __init__(self, ch):
        """
        :param ch: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PosEnc2D, self).__init__()
        self.org_ch = ch
        ch = int(np.ceil(ch / 4) * 2)
        self.ch = ch
        inv_freq = 1.0 / (10000 ** (torch.arange(0, ch, 2).float() / ch))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None

    def forward(self, x):
        """
        :param x: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(x.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cache is not None and self.cache.shape == x.shape:
            return self.cache

        self.cache = None
        bs, x_dim, y_dim, orig_ch = x.shape
        pos_x = torch.arange(x_dim, device=x.device).type(self.inv_freq.type())
        pos_y = torch.arange(y_dim, device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x_dim, y_dim, self.ch * 2), device=x.device).type(
            x.type()
        )
        emb[:, :, : self.ch] = emb_x
        emb[:, :, self.ch : 2 * self.ch] = emb_y

        self.cache = emb[None, :, :, :orig_ch].repeat(bs, 1, 1, 1)
        return self.cache


class PosEncPerm2D(nn.Module):
    def __init__(self, ch):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PosEncPerm2D, self).__init__()
        self.penc = PosEnc2D(ch)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        enc = self.penc(x)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_ch(self):
        return self.penc.org_ch


class PosEnc3D(nn.Module):
    def __init__(self, ch):
        """
        :param ch: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PosEnc3D, self).__init__()
        self.org_ch = ch
        ch = int(np.ceil(ch / 6) * 2)
        if ch % 2:
            ch += 1
        self.ch = ch
        inv_freq = 1.0 / (10000 ** (torch.arange(0, ch, 2).float() / ch))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None

    def forward(self, x):
        """
        :param x: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(x.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cache is not None and self.cache.shape == x.shape:
            return self.cache

        self.cache = None
        bs, x_dim, y_dim, z_dim, orig_ch = x.shape
        pos_x = torch.arange(x_dim, device=x.device).type(self.inv_freq.type())
        pos_y = torch.arange(y_dim, device=x.device).type(self.inv_freq.type())
        pos_z = torch.arange(z_dim, device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x_dim, y_dim, z_dim, self.ch * 3), device=x.device).type(
            x.type()
        )
        emb[:, :, :, : self.ch] = emb_x
        emb[:, :, :, self.ch : 2 * self.ch] = emb_y
        emb[:, :, :, 2 * self.ch :] = emb_z

        self.cache = emb[None, :, :, :, :orig_ch].repeat(bs, 1, 1, 1, 1)
        return self.cache


class PosEncPerm3D(nn.Module):
    def __init__(self, ch):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PosEncPerm3D, self).__init__()
        self.penc = PosEnc3D(ch)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        enc = self.penc(x)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_ch(self):
        return self.penc.org_ch


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param penc: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, x):
        """
        :param x: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(x)
        assert (
            x.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            x.size(), penc.size()
        )
        return x + penc


class SparsePosEnc2D(PosEnc2D):
    def __init__(self, ch, x_dim, y_dim, dev='cuda'):
        super(SparsePosEnc2D, self).__init__(ch)
        self.y, self.x = y_dim, x_dim
        self.fake = torch.zeros((1, x_dim, y_dim, ch), device=dev)

    def forward(self, coords):
        """
        :param coords: A list of list of coordinates (((x1, y1), (x2, y22), ... ), ... )
        :return: Positional Encoding Matrix summed to the original tensor
        """
        enc = super().forward(self.fake)
        enc = enc.permute(0, 3, 1, 2)
        idx = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(c) for c in coords], batch_first=True, padding_value=-1)
        idx = idx.unsqueeze(0).to(self.fake.device)
        assert self.x == self.y
        idx = (idx + 0.5) / self.x * 2 - 1
        idx = torch.flip(idx, (-1, ))
        return torch.nn.functional.grid_sample(enc, idx).squeeze().permute(2, 1, 0)


if __name__ == '__main__':
    pos = SparsePosEnc2D(10, 10, 20)
    pos([[0, 0], [0, 9], [1, 0], [9, 15]])