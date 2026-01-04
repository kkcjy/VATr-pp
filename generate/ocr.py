import os
import shutil

import cv2
import msgpack
import torch

from data.dataset import CollectionTextDataset, TextDataset, FolderDataset, FidDataset, get_dataset_path
from generate.writer import Writer
from util.text import get_generator


def generate_ocr(args):
    """
    Generate OCR training data. Words generated are from given text generator.
    """
    ds = CollectionTextDataset(
        args.dataset, 'files', TextDataset, 
        file_suffix=args.file_suffix, 
        num_examples=args.num_examples,
        collator_resolution=args.resolution, 
        validation=True
    )
    args.num_writers = ds.num_writers

    gen = Writer(args.checkpoint, args, only_generator=True)
    text_gen = get_generator(args)

    gen.generate_ocr(
        ds, 
        args.count, 
        interp_style=args.interp_styles, 
        out_dir=args.output, 
        text_gen=text_gen
    )


def generate_ocr_ref(args):
    """
    Generate OCR training data. Words generated are words from given dataset. Reference words are also saved.
    """
    ds = CollectionTextDataset(
        args.dataset, 'files', TextDataset, 
        file_suffix=args.file_suffix, 
        num_examples=args.num_examples,
        collator_resolution=args.resolution, 
        validation=True
    )

    args.num_writers = ds.num_writers
    gen = Writer(args.checkpoint, args, only_generator=True)

    gen.generate_ocr(
        ds, 
        args.count, 
        interp_style=args.interp_styles, 
        out_dir=args.output, 
        long_tail=args.long_tail
    )


def generate_ocr_msgpack(args):
    """
    Generate OCR dataset. Words generated are specified in given msgpack file
    """
    ds = FolderDataset(args.dataset_path)
    args.num_writers = 339

    if args.charset_file:
        charset = msgpack.load(open(args.charset_file, 'rb'), use_list=False, strict_map_key=False)
        args.alphabet = "".join(charset['char2idx'].keys())

    gen = Writer(args.checkpoint, args, only_generator=True)
    lines = msgpack.load(open(args.text_path, 'rb'), use_list=False)

    print(f"生成 {len(lines)} 到 {args.output}")

    for i, (fname, target) in enumerate(lines):
        if not os.path.exists(os.path.join(args.output, fname)):
            style = torch.unsqueeze(ds.sample_style()['simg'], dim=0).to(args.device)
            fake = gen.create_fake_sentence(style, target, at_once=True)

            cv2.imwrite(os.path.join(args.output, fname), fake)

    print("完成")