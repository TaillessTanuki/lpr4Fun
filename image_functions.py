import cv2

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

def ocr_with_angles(fpo, img_rgb):
    best, best_conf = None, -1
    for a in (-3, 0, 3):
        o = rotate(img_rgb, a)
        res = fpo.run(o)
        # if the API gives you a confidence score, pull it out here:
        conf = getattr(fpo, 'last_confidence', None) or 0
        if conf > best_conf:
            best_conf, best = conf, res[0]
    return best