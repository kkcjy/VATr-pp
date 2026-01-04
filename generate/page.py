import os

import cv2
import numpy as np
import torch

from data.dataset import CollectionTextDataset, TextDataset
from models.model import VATr
from util.loading import load_checkpoint, load_generator


def generate_page(args):
    args.output = 'vatr' if args.output is None else args.output

    args.vocab_size = len(args.alphabet)

    ds = CollectionTextDataset(
        args.dataset, 'files', TextDataset, 
        file_suffix=args.file_suffix, 
        num_examples=args.num_examples,
        collator_resolution=args.resolution
    )
    val_ds = CollectionTextDataset(
        args.dataset, 'files', TextDataset, 
        file_suffix=args.file_suffix, 
        num_examples=args.num_examples,
        collator_resolution=args.resolution, 
        validation=True
    )

    args.num_writers = ds.num_writers

    model = VATr(args)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = load_generator(model, ckpt)

    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True, 
        drop_last=True,
        collate_fn=ds.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True, 
        drop_last=True,
        collate_fn=val_ds.collate_fn)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    model.eval()
    with torch.no_grad():
        page = model._generate_page(train_batch['simg'].to(args.device), train_batch['swids'])
        val_page = model._generate_page(val_batch['simg'].to(args.device), val_batch['swids'])

    cv2.imwrite(os.path.join("saved_images", "pages", f"{args.output}_train.png"), (page * 255).astype(np.uint8))
    cv2.imwrite(os.path.join("saved_images", "pages", f"{args.output}_val.png"), (val_page * 255).astype(np.uint8))