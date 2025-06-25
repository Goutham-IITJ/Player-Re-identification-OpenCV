import cv2
from yolo_detector import YOLODetector
from feature_extractor import FeatureExtractor
from reid_tracker import ReIDTracker
from helpers import crop_bbox

video_path = "15sec_input_720p.mp4"
output_path = "outputs/tracked_players_new.mp4"

detector = YOLODetector("best.pt")
extractor = FeatureExtractor(device="cpu")
tracker = ReIDTracker(similarity_threshold=0.85)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    raw_dets = detector.detect(frame)

    detections = []
    for det in raw_dets:
        cropped = crop_bbox(frame, det["bbox"])
        if cropped.size == 0: continue
        feature = extractor.extract(cropped)
        det["feature"] = feature
        detections.append(det)

    tracked = tracker.update(detections)

    for obj in tracked:
        x, y = map(int, obj.estimate[0])
        track_id = obj.id
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame, f"Player {track_id}", (x - 10, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()
print("\n✅ Final tracked video saved to", output_path)
