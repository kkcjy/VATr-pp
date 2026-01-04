import math
import os

import cv2
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np


def gauss(x, sigma=1.0):
    return (1.0 / math.sqrt(2.0 * math.pi) * sigma) * math.exp(-x**2 / (2.0 * sigma**2))


class UnifontMod(torch.nn.Module):
    def __init__(self, out_dim, alph, dev='cuda', in_type='unifont', proj='linear'):
        super(UnifontMod, self).__init__()
        self.proj_type = proj
        self.dev = dev
        self.alph = alph
        self.syms = self.get_syms('unifont')
        self.syms_repr = self.get_syms(in_type)

        if proj == 'linear':
            self.linear = torch.nn.Linear(self.syms_repr.shape[1], out_dim)
        else:
            self.linear = torch.nn.Identity()

    def get_syms(self, in_type):
        with open(f"files/{in_type}.pickle", "rb") as f:
            syms = pickle.load(f)

        all_syms = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in syms}
        syms_list = []
        for char in self.alph:
            im = all_syms[ord(char)]
            im = im.flatten()
            syms_list.append(im)

        syms_list.insert(0, np.zeros_like(syms_list[0]))
        syms_arr = np.stack(syms_list)
        return torch.from_numpy(syms_arr).float().to(self.dev)

    def forward(self, QR):
        if self.proj_type != 'cnn':
            return self.linear(self.syms_repr[QR])
        else:
            res = []
            syms = self.syms_repr[QR]
            for b in range(QR.size(0)):
                res.append(self.linear(torch.unsqueeze(syms[b], dim=1)))

            return torch.stack(res)


class LearnMod(torch.nn.Module):
    def __init__(self, out_dim, dev='cuda'):
        super(LearnMod, self).__init__()
        self.dev = dev
        self.param = torch.nn.Parameter(torch.zeros(1, 1, 256, device=dev))
        self.linear = torch.nn.Linear(256, out_dim)

    def forward(self, QR):
        return self.linear(self.param).repeat((QR.shape[0], 1, 1))


if __name__ == "__main__":
    mod = UnifontMod(512, "bluuuuurp", 'cpu', proj='cnn')