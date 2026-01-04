import numpy as np


def stack_lines(lines: list, gap: int = 6):
    w = max([img.shape[1] for img in lines])
    h = (lines[0].shape[0] + gap) * len(lines)

    stacked = np.ones((h, w)) * 255

    y = 0
    for img in lines:
        h_img, w_img = img.shape[:2]
        stacked[y:y + h_img, :w_img] = img
        y += h_img + gap

    return stacked