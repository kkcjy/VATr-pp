import os.path
import pickle
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np
import math
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from data.dataset import CollectionTextDataset, TextDataset


def to_cv2(batch: torch.Tensor):
    imgs = []

    for img in batch:
        img = img.detach().cpu().numpy()
        img = (img + 1.0) / 2.0
        imgs.append(np.squeeze(img))

    return imgs


class RandMorph(torch.nn.Module):
    def __init__(self, max_sz: 5, max_iter = 1, op = cv2.MORPH_ERODE):
        super().__init__()
        self.elems = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE]
        self.max_sz = max_sz
        self.max_iter = max_iter
        self.op = op

    def forward(self, x):
        dev = x.device

        imgs = to_cv2(x)

        res = []

        sz = random.randint(1, self.max_sz)
        kernel = cv2.getStructuringElement(random.choice(self.elems), (sz, sz))

        for img in imgs:
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            morph = cv2.morphologyEx(img, op=self.op, kernel=kernel, iterations=random.randint(1, self.max_iter))
            morph = cv2.resize(morph, (img.shape[1] // 2, img.shape[0] // 2))
            morph = morph * 2.0 - 1.0

            res.append(torch.Tensor(morph))

        return torch.unsqueeze(torch.stack(res).to(dev), dim=1)


def gauss_noise(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 0.075

    out = img + sigma * (torch.randn_like(img) - 0.5)

    out = torch.clamp(out, -1.0, 1.0)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


def word_w(img: torch.Tensor) -> int:
    idxs = torch.where((img < 0).int())[2]
    idx = torch.max(idxs) if len(idxs) > 0 else img.size(-1)

    return idx


class Down(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(0.0, scale=(0.8, 1.0), interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=1.0),
            torchvision.transforms.GaussianBlur(3, sigma=0.3)
        ])

    def forward(self, x):
        return self.aug(x)


class OCRAug(torch.nn.Module):
    def __init__(self, prob: float = 0.5, n: int = 2):
        super().__init__()
        self.prob = prob
        self.n = n

        interp = torchvision.transforms.InterpolationMode.NEAREST
        fill = 1.0

        self.augs = [
            torchvision.transforms.RandomRotation(3.0, interpolation=interp, fill=fill),
            torchvision.transforms.RandomAffine(0.0, translate=(0.05, 0.05), interpolation=interp, fill=fill),
            Down(),
            torchvision.transforms.ElasticTransform(alpha=10.0, sigma=7.0, fill=fill, interpolation=interp),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),
            torchvision.transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            gauss_noise,
            RandMorph(max_sz=4, max_iter=2, op=cv2.MORPH_ERODE),
            RandMorph(max_sz=2, max_iter=1, op=cv2.MORPH_DILATE)
        ]

    def forward(self, x):
        if random.uniform(0.0, 1.0) > self.prob:
            return x

        augs = random.choices(self.augs, k=self.n)

        for aug in augs:
            x = aug(x)

        return x


class WordCrop(torch.nn.Module, ABC):
    def __init__(self, pad: bool = False):
        super().__init__()
        self.pad = pad
        self.pad_trans = torchvision.transforms.Pad([2, 2, 2, 2], 1.0)

    @abstractmethod
    def curr_w(self):
        pass

    @abstractmethod
    def update(self, epoch: int):
        pass

    def forward(self, imgs):
        assert len(imgs.size()) == 4 and imgs.size(1) == 1, "Augmentation works on batches of one channel images"

        if self.pad:
            imgs = self.pad_trans(imgs)

        res = []
        w = self.curr_w()

        for img in imgs:
            idx = word_w(img)
            max_idx = max(min(idx - w // 2, img.size(2) - w), 0)
            start = random.randint(0, max_idx)

            res.append(F.crop(img, 0, start, img.size(1), min(w, img.size(2))))

        return torch.stack(res)


class StaticCrop(WordCrop):
    def __init__(self, w: int, pad: bool = False):
        super().__init__(pad=pad)
        self.w = w

    def curr_w(self):
        return int(self.w)

    def update(self, epoch: int):
        pass


class RandCrop(WordCrop):
    def __init__(self, min_w: int, max_w: int, pad: bool = False):
        super().__init__(pad)

        self.min_w = min_w
        self.max_w = max_w

        self.curr_w_val = random.randint(self.min_w, self.max_w)

    def update(self, epoch: int):
        self.curr_w_val = random.randint(self.min_w, self.max_w)

    def curr_w(self):
        return self.curr_w_val


class FullCrop(torch.nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.w = w
        self.h = 32
        self.pad = torchvision.transforms.Pad([6, 6, 6, 6], 1.0)

    def curr_w(self):
        return self.w

    def forward(self, imgs):
        assert len(imgs.size()) == 4 and imgs.size(1) == 1, "Augmentation works on batches of one channel images"
        imgs = self.pad(imgs)

        res = []

        for img in imgs:
            idx = word_w(img)
            max_idx = max(min(idx - self.w // 2, img.size(2) - self.w), 0)

            start_w = random.randint(0, max_idx)
            start_h = random.randint(0, img.size(1) - self.h)

            res.append(F.crop(img, start_h, start_w, self.h, min(self.w, img.size(2))))

        return torch.stack(res)


class ProgCrop(WordCrop):
    def __init__(self, w: int, warmup: int, start_w: int = 128, pad: bool = False):
        super().__init__(pad=pad)
        self.tgt_w = w
        self.warmup = warmup
        self.start_w = start_w
        self.curr_w_val = float(start_w)

    def update(self, epoch: int):
        val = self.start_w - ((self.start_w - self.tgt_w) / self.warmup) * epoch
        self.curr_w_val = max(val, self.tgt_w)

    def curr_w(self):
        return int(round(self.curr_w_val))


class CycleCrop(WordCrop):
    def __init__(self, w: int, cycle: int, start_w: int = 128, pad: bool = False):
        super().__init__(pad=pad)

        self.tgt_w = w
        self.start_w = start_w
        self.curr_w_val = float(start_w)
        self.cycle = float(cycle)

    def update(self, epoch: int):
        val = (math.cos((float(epoch) * 2 * math.pi) / self.cycle) + 1) * ((self.start_w - self.tgt_w) / 2) + self.tgt_w
        self.curr_w_val = val

    def curr_w(self):
        return int(round(self.curr_w_val))


class ResizeH(torch.nn.Module):
    def __init__(self, tgt_h: int):
        super().__init__()
        self.tgt_h = tgt_h

    def forward(self, x):
        w, h = F.get_image_size(x)
        scale = self.tgt_h / h

        return F.resize(x, [int(h * scale), int(w * scale)])



def show_crops():
    with open("../files/IAM-32-pa.pickle", 'rb') as f:
        data = pickle.load(f)

    for auth in data['train'].keys():
        for img in data['train'][auth]:
            img = torch.Tensor(np.expand_dims(np.expand_dims(np.array(img['img']), 0), 0))

            aug = torchvision.transforms.Compose([
                ResizeH(32),
                FullCrop(128)
            ])

            batch = aug(img)

            batch = batch.detach().cpu().numpy()
            res = [np.squeeze(im) for im in batch]

            f, ax = plt.subplots(1, len(res))

            for i in range(len(res)):
                ax.imshow(res[i])

            plt.show()


if __name__ == "__main__":
    ds = CollectionTextDataset(
        'IAM', '../files', TextDataset, file_suffix='pa', num_examples=15,
        collator_resolution=16, min_virtual_size=339, validation=False, debug=False
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        pin_memory=True, drop_last=True,
        collate_fn=ds.collate_fn)

    aug = OCRAug(n=3, prob=1.0)

    out_dir = r"C:\Users\bramv\Documents\Werk\Research\Unimore\VATr\VATr_ext\saved_images\debug\ocr_aug"

    img_cnt = 0

    for batch in loader:
        for i in range(5):
            auged = aug(batch["img"])

            img = np.squeeze((auged[0].detach().cpu().numpy() + 1.0) / 2.0)

            img = (img * 255.0).astype(np.uint8)

            print(cv2.imwrite(os.path.join(out_dir, f"{img_cnt}_{i}.png"), img))

        img = np.squeeze((batch["img"][0].detach().cpu().numpy() + 1.0) / 2.0)
        img = (img * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{img_cnt}.png"), img)

        if img_cnt > 5:
            break

        img_cnt+=1