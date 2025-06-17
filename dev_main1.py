import easyocr
import cv2
import numpy as np
from ultralytics import YOLO
from typo import correct_prefix, extract_plate_core
from color_plate import detect_plate_type


image_path = "platmobil15.jpg"
image_downloaded = "downloaded.jpg"
plate_detector = YOLO('license_plate_detector.pt')

# Detect license plate
license_plates = plate_detector(image_path)[0]
if not license_plates.boxes.data.tolist():
    print("message: No license plate detected.")
    exit()
else:
    print("message: License plates detected")

# Get only the first license plate and crop it
license_plate = license_plates.boxes.data.tolist()[0]
x1, y1, x2, y2, score, class_id = license_plate
image_array = cv2.imread(image_path)
license_plate_cropped = image_array[int(y1):int(y2), int(x1):int(x2), :]
cv2.imwrite(image_downloaded, license_plate_cropped)

# If the plate is red or not

plate_type = detect_plate_type(license_plate_cropped)
if plate_type == 'red':
    print(f"the plate is red")
    img = license_plate_cropped.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 180, 255,
                          cv2.THRESH_BINARY_INV)
    license_plate_cropped = thresh.copy()


# Read license plate
ocr = easyocr.Reader(["en"])
results = ocr.readtext(license_plate_cropped, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

if not results:
    print("No result")
    exit()


print(f"result: {results}")
sorted_results = sorted(results, key=lambda r: r[0][0][0])
print(f"sorted_result: {sorted_results}")
bbox_image = license_plate_cropped.copy()

plate_parts_y_bottom = []
upper_text_y_bottom = ""
plate_parts_height = []
upper_text_height = ""
base_y_bottom = sorted_results[0][0][2][1]
y_bottom_tolerance = 20
plate_parts_base_y_bottom = []
first_box = sorted_results[0][0]
first_height = abs(first_box[2][1] - first_box[0][1])
char_height_threshold = 0.6 * first_height
plate_parts_first_height = []
upper_text_first_height =""

for bbox, text, conf in results:
    x_left = bbox[0][0]
    y_top = bbox[0][1]
    y_bottom = bbox[2][1]
    print(f"y_bottom: {y_bottom}")
    height = y_bottom - y_top
    print(f"height: {height}")
    print(f"base_y_bottom: {base_y_bottom}")
    print(f"first_height: {first_height}")

    # Draw bbox to pic
    points = [(int(x), int(y)) for x, y in bbox]
    cv2.polylines(bbox_image, [np.array(points)],isClosed=True, color=(255,0,0),thickness=2)
    cv2.putText(bbox_image,text,(points[0][0], points[0][1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # Use y_bottom of each char as the parameter (y_bottom)
    if y_bottom < 100 :
        clean_text_y_bottom = "".join(text.split())
        if clean_text_y_bottom.isalnum():  # Alphanumeric check
            plate_parts_y_bottom.append((x_left, clean_text_y_bottom))
            print(f"plate_parts_y_bottom {plate_parts_y_bottom}")

    # Use the height of the char as the parameter (height)
    if height > 5 :
        clean_text_height = "".join(text.split())
        if clean_text_height.isalnum():
            plate_parts_height.append((x_left, clean_text_height))
            print(f"plate_parts_height {plate_parts_height}")

    # Use the y_bottom of the first char (base_y_bottom) as the parameter
    clean_text_base_y_bottom = "".join(text.split())
    if abs(y_bottom - base_y_bottom) <= y_bottom_tolerance and clean_text_base_y_bottom.isalnum():
        plate_parts_base_y_bottom.append((x_left, clean_text_base_y_bottom))
        print(f"plate_parts_base_y_bottom {plate_parts_base_y_bottom}")

    # Use the height of the first char as the parameter (first_height)
    char_height = abs(bbox[2][1]-bbox[0][1])
    if char_height >= char_height_threshold and text.strip().isalnum():
        plate_parts_first_height.append((bbox[0][0], text.strip()))
        print(f"plate_parts_first_height {plate_parts_first_height}")


# Download image with bbox
cv2.imwrite("bboxes_image.jpg", bbox_image)

# Sort by vertical position, then join all parts (y_bottom)
plate_parts_y_bottom.sort(key=lambda x: x[0])
upper_text_y_bottom = "".join([text for _, text in plate_parts_y_bottom])
print(f"raw license plate y_bottom {upper_text_y_bottom}")
# Cut off everything after the last letter
last_alpha_index_y_bottom = max((i for i, c in enumerate(upper_text_y_bottom) if c.isalpha()), default=-1)
license_plate_final_y_bottom = upper_text_y_bottom[:last_alpha_index_y_bottom + 1] if last_alpha_index_y_bottom != -1 else upper_text_y_bottom
corrected_license_plate_y_bottom = correct_prefix(license_plate_final_y_bottom)
print(f"final license plate y_bottom {corrected_license_plate_y_bottom}")

# Sort by vertical position, then join all parts (height)
plate_parts_height.sort(key=lambda x: x[0])
upper_text_height = "".join([text for _, text in plate_parts_height])
print(f"raw license plate height {upper_text_height}")
# Cut off everything after the last letter
last_alpha_index_height = max((i for i, c in enumerate(upper_text_height) if c.isalpha()), default=-1)
license_plate_final_height = upper_text_height[:last_alpha_index_height + 1] if last_alpha_index_height != -1 else upper_text_height
corrected_license_plate_final_height = correct_prefix(license_plate_final_height)
print(f"final license plate y_bottom {corrected_license_plate_final_height}")

# Sort by vertical position, then join all parts (base_y_bottom)
plate_parts_base_y_bottom.sort(key=lambda x: x[0])
upper_text_base_y_bottom = "".join([text for _, text in plate_parts_base_y_bottom])
print(f"raw license plate base_y_bottom {upper_text_base_y_bottom}")
corrected_license_plate_base_y_bottom = correct_prefix(upper_text_base_y_bottom)
print(f"final license base_y_bottom {corrected_license_plate_base_y_bottom}")

# Sort by vertical position, then join all part (first_height)
plate_parts_first_height.sort(key= lambda x: x[0])
upper_text_first_height = "".join([text for _, text in plate_parts_first_height])
print(f"raw plate_parts_first_height {upper_text_first_height}")
corrected_license_plate_first_height = correct_prefix(upper_text_first_height)
print(f"license first_height correct prefix {corrected_license_plate_first_height}")
extract_corrected_license_plate_first_height = extract_plate_core(corrected_license_plate_first_height)
print(f"final license first_height {extract_corrected_license_plate_first_height}")


