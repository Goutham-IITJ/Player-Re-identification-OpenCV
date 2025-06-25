import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker
import numpy as np
import os

# Load YOLOv8 model
model = YOLO("best.pt")

# Class ID mapping
class_map = {
    0: 'ball',
    1: 'goalkeeper',
    2: 'player',
    3: 'referee'
}

# Initialize tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Input/output
video_path = "15sec_input_720p.mp4"
output_dir = "outputs"
output_path = os.path.join(output_dir, "tracked_players_final.avi")  # Using .avi for compatibility

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]
    detections = []

    # Loop over detected boxes
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        if cls_id in class_map and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Add detection point
            detections.append(
                Detection(points=np.array([[cx, cy]]), scores=np.array([conf]))
            )

    # Update tracker
    tracked_objects = tracker.update(detections=detections)

    # Draw tracked objects
    for tracked in tracked_objects:
        track_id = tracked.id
        x, y = map(int, tracked.estimate[0])
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(frame, f"ID {track_id}", (x - 10, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Tracked video saved to {output_path}")
