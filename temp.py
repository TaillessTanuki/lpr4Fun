import cv2


license_plates = lpr_model(image)[0]
if not license_plates.boxes.data.tolist():
    return {"message": "No license plates detected."}

results_response = []  # <-- rename to avoid conflict

# Get only the first plate
license_plate = license_plates.boxes.data.tolist()[0]
x1, y1, x2, y2, score, class_id = license_plate

# Crop the license plate
license_plate_cropped = image[int(y1):int(y2), int(x1):int(x2), :]
license_plate_cropped_resized = cv2.resize(license_plate_cropped, None, fx=2, fy=2,
                                           interpolation=cv2.INTER_LINEAR)
license_plate_cropped_resized_gray = cv2.cvtColor(license_plate_cropped_resized, cv2.COLOR_BGR2GRAY)

# OCR
reader = easyocr.Reader(['en'], gpu=True)
ocr_results = reader.readtext(license_plate_cropped_resized_gray)

# Sort OCR results left to right
ocr_results.sort(key=lambda x: x[0][0][0])
plate_text = ''.join([text.upper().replace(' ', '') for (_, text, _) in ocr_results])
print("Raw Plate:", plate_text)
# plate_text = correct_ocr_plate(plate_text)
# print("Corrected Plate:", plate_text)

# Debug
cv2.imwrite(f"debug_plate_{int(x1)}_{int(y1)}.jpg", license_plate_cropped)
print("Final Plate:", plate_text)

# Add to result
if plate_text:
    return {
        "vehicle_type": vehicle_type,
        "text": str(plate_text)
    }

else:
    return {
        "message": "No text detected"
    }

