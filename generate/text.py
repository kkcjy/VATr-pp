from pathlib import Path

import cv2

from generate.writer import Writer


def generate_text(args):
    if args.text_path:
        with open(args.text_path, 'r') as f:
            args.text = f.read()
    
    args.text = args.text.splitlines()
    args.output = 'files/output.png' if not args.output else args.output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.num_writers = 0

    w = Writer(args.checkpoint, args, only_generator=True)
    w.set_style_folder(args.style_folder)
    
    imgs = w.generate(args.text, args.align)
    for i, img in enumerate(imgs):
        dst = out_path.parent / (out_path.stem + f'_{i:03d}' + out_path.suffix)
        cv2.imwrite(str(dst), img)
    
    print('完成')