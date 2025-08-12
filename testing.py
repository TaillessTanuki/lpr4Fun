# import time
#
# start_baking = time.time()
# time.sleep(90)
# now_baking = time.time()
#
# elapse_baking = now_baking - start_baking
# minute_format = elapse_baking // 60
# second_format = elapse_baking % 60
# formatted_time = f"{int(minute_format):02d}:{int(second_format):02d}"
# print(f'elapse_baking: {formatted_time}')


# from realesrgan import RealESRGANer
#
# # real-esrganer
# sr_re = RealESRGANer(scale=4, model_path='RealESRGAN_x4.pth', gpu_id=-1)
# plate_sr_re, _ = sr_re.enhance(license_plate_cropped, outscale=4)



import easyocr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typo import correct_prefix, extract_plate_core
from color_plate import detect_plate_type
from fast_plate_ocr import LicensePlateRecognizer
from cv2 import dnn_superres
from realesrgan import RealESRGANer
from image_functions import rotate, ocr_with_angles


image_path = "lprimage1.png"
image_downloaded = "downloaded.jpg"
image_downloaded_2 = "downloaded2.jpg"
image_downloaded_preprocessed = "downloaded_preprocessed.jpg"
plate_detector = YOLO('license_plate_detector.pt')
# Model selection

## fast-plate-ocr
## cct-s-v1-global-model
## cct-xs-v1-global-model
fpo_lpr = LicensePlateRecognizer("cct-s-v1-global-model")

## Learned RSR
rsr = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4.pth',
    gpu_id=-1
)

# 1. Detect & crop the plate
results = plate_detector(image_path)[0]
if not results.boxes.data.tolist():
    print("No plate detected."); exit()
x1, y1, x2, y2, *_ = results.boxes.data.tolist()[0]
orig = cv2.imread(image_path)
crop = orig[int(y1):int(y2), int(x1):int(x2)]

# save your raw crop
cv2.imwrite(image_downloaded, crop)

# 2. EDSR ×4 (OpenCV DNN)
ed = dnn_superres.DnnSuperResImpl_create()
ed.readModel('EDSR_x4.pb')
ed.setModel('edsr', 4)
crop_ed = ed.upsample(crop)
cv2.imwrite(image_downloaded_2, crop_ed)

# 3. Real-ESRGANer ×4
rsr = RealESRGANer(scale=4, model_path='RealESRGAN_x4.pth', gpu_id=-1)
crop_rsr, _ = rsr.enhance(crop, outscale=4)   # note: feed the raw crop here

# 4. Choose which SR output to feed into your light-touch pipeline:
#    e.g. use crop_rsr if you want the learned SR, or crop_ed for DNN SR

sr_input = crop_rsr  # or crop_ed

# 5. Gamma / unsharp / threshold pipeline
h, w = sr_input.shape[:2]
up = cv2.resize(sr_input, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
gamma = 1.4
inv = 1/gamma
lut = np.array([(i/255.0)**inv * 255 for i in range(256)], np.uint8)
gamma_img = cv2.LUT(up, lut)
gauss = cv2.GaussianBlur(gamma_img, (0,0), sigmaX=1.0)
sharp = cv2.addWeighted(gamma_img, 1.4, gauss, -0.4, 0)
gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=15, C=2
)
mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
final = cv2.resize(mask, (128, 64), interpolation=cv2.INTER_CUBIC)
final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

# save your final preprocessed image
cv2.imwrite(image_downloaded_preprocessed, final_rgb)

# 6. OCR passes
fpo = LicensePlateRecognizer("cct-s-v1-global-model")
# single-pass
print("OCR on learned SR preproc:", fpo.run(final_rgb))
# ensemble on rotations
print("Rot-ensemble:", ocr_with_angles(fpo, final_rgb))
