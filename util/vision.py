import numpy as np
import cv2


def detect_text_bounds(img: np.array) -> (int, int):
    """
    Find the lower and upper bounding lines in an image of a word
    """
    if len(img.shape) >= 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) >= 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)

    _, thresh = cv2.threshold(img, 0.8, 1, cv2.THRESH_BINARY_INV)

    sums = np.sum(thresh, axis=1).astype(float)
    sums = np.convolve(sums, np.ones(5) / 5, mode='same')

    sums_d = np.diff(sums)

    std_factor = 0.5
    min_th = np.mean(sums_d[sums_d <= 0]) - std_factor * np.std(sums_d[sums_d <= 0])
    bot_idx = np.max(np.where(sums_d < min_th))

    max_th = np.mean(sums_d[sums_d >= 0]) + std_factor * np.std(sums_d[sums_d >= 0])
    top_idx = np.min(np.where(sums_d > max_th))

    return bot_idx, top_idx


def dist(p1, p2) -> float:
    return np.linalg.norm(p2 - p1)


def crop(img: np.array, ratio: float = None, px: int = None) -> np.array:
    assert ratio is not None or px is not None, "Please specify either pixels or a ratio to crop"

    w, h = img.shape[:2]

    if ratio is not None:
        w_crop = int(ratio * w)
        h_crop = int(ratio * h)
    else:
        w_crop = px
        h_crop = px

    return img[h_crop:h-h_crop, w_crop:w-w_crop]


def find_target_points(tl, tr, bl, br):
    max_w = max(int(dist(br, bl)), int(dist(tr, tl)))
    max_h = max(int(dist(tr, br)), int(dist(tl, bl)))
    dst_corners = [[0, 0], [max_w, 0], [max_w, max_h], [0, max_h]]

    return order_points(dst_corners)


def order_points(pts: np.array) -> tuple:
    """
    inspired by: https://learnopencv.com/automatic-document-scanner-using-opencv/
    """
    s = np.sum(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return tl, tr, bl, br


def get_page(img: np.array) -> np.array:
    """
    inspired by: https://github.com/Kakaranish/OpenCV-paper-detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)

    edges = cv2.Canny(gray, 30, 50, 3)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_peri = 0
    max_cnt = None
    for cnt in cnts:
        cnt = np.array(cnt)
        peri = cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if peri > max_peri and cv2.isContourConvex(cnt_approx) and len(cnt_approx) == 4:
            max_peri = peri
            max_cnt = cnt_approx

    if max_cnt is not None:
        max_cnt = np.squeeze(max_cnt)
        pts = order_points(max_cnt)

        tgt_pts = find_target_points(*pts)
        M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(tgt_pts))
        warped = cv2.warpPerspective(img, M, (tgt_pts[3][0], tgt_pts[3][1]), flags=cv2.INTER_LINEAR)
        warped = crop(warped, px=10)
        return warped

    return img


def get_words(page: np.array, dil_size: int = 3):
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 125, 1, cv2.THRESH_BINARY_INV)

    dil_size = dil_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_size + 1, 2 * dil_size + 1),
                                       (dil_size, dil_size))
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    words = []
    boxes = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if ratio <= 0.1 or ratio >= 10.0:
            continue
        boxes.append([x, y, w, h])
        words.append(page[y:y+h, x:x+w])

    return words, boxes