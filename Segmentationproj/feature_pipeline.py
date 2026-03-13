import cv2
import numpy as np

IMG_SIZE = 128
MIN_LEAF_AREA_RATIO = 0.05


def _create_hog_descriptor():
    return cv2.HOGDescriptor(
        _winSize=(IMG_SIZE, IMG_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


HOG_DESCRIPTOR = _create_hog_descriptor()


def isolate_leaf(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([95, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green_mask, yellow_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < img_bgr.shape[0] * img_bgr.shape[1] * MIN_LEAF_AREA_RATIO:
        return img_bgr, np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    leaf_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest)
    cropped_img = img_bgr[y:y + h, x:x + w]
    cropped_mask = leaf_mask[y:y + h, x:x + w]

    isolated = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
    white_background = np.full_like(isolated, 255)
    white_background[cropped_mask > 0] = isolated[cropped_mask > 0]

    return white_background, cropped_mask


def extract_lbp_histogram(gray_img):
    center = gray_img[1:-1, 1:-1]
    neighbors = [
        gray_img[:-2, :-2],
        gray_img[:-2, 1:-1],
        gray_img[:-2, 2:],
        gray_img[1:-1, 2:],
        gray_img[2:, 2:],
        gray_img[2:, 1:-1],
        gray_img[2:, :-2],
        gray_img[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center) << bit).astype(np.uint8)

    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum() + 1e-8
    return hist


def extract_color_features(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    stats = []
    for color_img in (img_bgr, img_hsv, img_lab):
        stats.extend(color_img.mean(axis=(0, 1)))
        stats.extend(color_img.std(axis=(0, 1)))

    hsv_hist = cv2.calcHist(
        [img_hsv], [0, 1, 2],
        None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    return np.concatenate([np.asarray(stats, dtype=np.float32), hsv_hist])


def extract_shape_features(gray_img):
    edges = cv2.Canny(gray_img, 80, 160)
    edge_density = np.array([np.count_nonzero(edges) / edges.size], dtype=np.float32)

    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.HuMoments(cv2.moments(binary)).flatten()
    moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-8)

    return np.concatenate([edge_density, moments.astype(np.float32)])


def extract_features(img_bgr):
    isolated_leaf, _ = isolate_leaf(img_bgr)
    resized = cv2.resize(isolated_leaf, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hog_features = HOG_DESCRIPTOR.compute(gray).flatten()
    lbp_hist = extract_lbp_histogram(gray)
    color_features = extract_color_features(resized)
    shape_features = extract_shape_features(gray)

    return np.concatenate(
        [
            hog_features.astype(np.float32),
            lbp_hist.astype(np.float32),
            color_features.astype(np.float32),
            shape_features.astype(np.float32),
        ]
    )
