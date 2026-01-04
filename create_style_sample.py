import os
import argparse

import cv2
from util.vision import get_page, get_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True, default='files/style_samples/00')

    args = parser.parse_args()

    img = cv2.imread(args.input_image)
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    page = get_page(img)
    words, _ = get_words(page)

    out_dir = args.output_folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, word in enumerate(words):
        cv2.imwrite(os.path.join(out_dir, f"word{i}.png"), word)