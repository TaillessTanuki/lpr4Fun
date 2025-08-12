import cv2
import numpy as np
import time
import base64

# Import SORT tracker and your detection/utility functions
from sort.sort import Sort
from image_function import detection, save_vehicle_image, detect_license_plate

# Path to the input video file (configure here)
VIDEO_PATH = "jalan-raya-lintas-timur.mp4"

# Map YOLO class IDs to vehicle types
vehicle_types = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# IoU helper function
def iou(a, b):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return

    tracker = Sort()
    last_sent = {}  # track_id -> timestamp

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 1. Detect vehicles
        dets_raw = detection(frame)[0].boxes.data.tolist()
        dets, class_ids = [], []
        for x1, y1, x2, y2, score, class_id in dets_raw:
            if int(class_id) in vehicle_types:
                dets.append([x1, y1, x2, y2, score])
                class_ids.append(int(class_id))
        dets = np.array(dets)

        # 2. Convert to NumPy, ensuring shape (N,5)
        if len(dets) == 0:
            dets_np = np.empty((0, 5), dtype=float)
        else:
            dets_np = np.array(dets, dtype=float)

        # 2. Track
        tracks = tracker.update(dets_np)

        # 3. Draw all tracks on frame (bbox + track_id)
        for x1, y1, x2, y2, track_id in tracks:
            # Draw bounding box
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            # Draw track_id label
            cv2.putText(frame, f"ID:{int(track_id)}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. Process each track for payload (with 5s throttle)
        for x1, y1, x2, y2, track_id in tracks:
            now = time.time()
            if now - last_sent.get(track_id, 0) < 5.0:
                continue

            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

            # Match track back to detection to get class_id
            best_iou, best_idx = 0, None
            for idx, det in enumerate(dets):
                det_bbox = [det[0], det[1], det[2] - det[0], det[3] - det[1]]
                score_iou = iou(det_bbox, bbox)
                if score_iou > best_iou:
                    best_iou, best_idx = score_iou, idx
            if best_iou < 0.3:
                continue

            class_id = class_ids[best_idx]
            vehicle_type = vehicle_types.get(class_id, 'unknown')

            # Crop & validate
            vehicle_image = save_vehicle_image(frame, bbox, track_id, vehicle_type)
            if vehicle_image is None:
                continue

            # License plate detection
            if not detect_license_plate(vehicle_image, track_id):
                continue

            # Build & print payload
            _, buf = cv2.imencode('.jpg', vehicle_image)
            img_b64 = base64.b64encode(buf).decode('utf-8')
            payload = {
                'id': track_id,
                'bbox': bbox,
                'image': "img_b64",
                'video_source': VIDEO_PATH,
                'vehicle_type': vehicle_type,
                'timestamp': now
            }
            print(f"Payload for track {track_id}, bbox={bbox}: {payload}")
            last_sent[track_id] = now

        # 5. Display frame
        cv2.imshow('Vehicle Tracking Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
