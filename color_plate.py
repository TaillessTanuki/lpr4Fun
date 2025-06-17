import cv2
import numpy as np

def detect_plate_type(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv[:, :, 0])
    if avg_color < 10 or avg_color > 160:  # RED hue wraps around in HSV
        return 'red'
    elif np.mean(img) > 170:
        return 'white'
    else:
        return 'black'


def preprocess_by_plate_type(img):
    plate_type = detect_plate_type(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if plate_type == 'white':
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 4)
    elif plate_type == 'black':
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif plate_type == 'red':
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 180, 255,
                                  cv2.THRESH_BINARY_INV)
    else:
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    return thresh