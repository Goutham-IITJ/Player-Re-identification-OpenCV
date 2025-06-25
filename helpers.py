import cv2
import numpy as np
# helpers.py
def crop_bbox(image, bbox, size=(128, 256)):
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Resize with padding to keep aspect ratio
    h, w = crop.shape[:2]
    scale = min(size[0]/w, size[1]/h)
    resized = cv2.resize(crop, (int(w*scale), int(h*scale)))
    padded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pad_y = (size[1] - resized.shape[0]) // 2
    pad_x = (size[0] - resized.shape[1]) // 2
    padded[pad_y:pad_y+resized.shape[0], pad_x:pad_x+resized.shape[1]] = resized

    return padded
