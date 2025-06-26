import cv2
import os
from yolo_detector import YOLODetector
from feature_extractor import FeatureExtractor
from reid_tracker import ReIDTracker
from helpers import crop_bbox

def main():
    video_path = "15sec_input_720p.mp4"
    output_path = "outputs/tracked_players_new_v2.mp4"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize components with better parameters
    detector = YOLODetector("best.pt", device="cpu")
    extractor = FeatureExtractor(device="cpu")
    tracker = ReIDTracker(similarity_threshold=0.8, max_disappeared=60)  # More lenient threshold, longer memory
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use a more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    print(f"🎥 Processing {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players
        raw_dets = detector.detect(frame)
        
        # Filter detections by confidence and size
        valid_detections = []
        for det in raw_dets:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Filter out very small detections (likely false positives)
            width_bbox = x2 - x1
            height_bbox = y2 - y1
            if width_bbox < 30 or height_bbox < 50:  # Minimum player size
                continue
                
            # Ensure bbox is within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Crop and extract features
            cropped = crop_bbox(frame, (x1, y1, x2, y2))
            if cropped.size == 0:
                continue
                
            feature = extractor.extract(cropped)
            if feature is not None:
                det["bbox"] = (x1, y1, x2, y2)  # Update with clipped bbox
                det["feature"] = feature
                valid_detections.append(det)
        
        # Update tracker
        tracked_objects = tracker.update(valid_detections)
        
        # Draw tracking results
        for obj in tracked_objects:
            if len(obj.estimate) > 0:
                # Get center point
                center_x, center_y = map(int, obj.estimate[0])
                
                # Ensure center is within frame bounds
                center_x = max(10, min(width - 10, center_x))
                center_y = max(20, min(height - 10, center_y))
                
                track_id = obj.id
                
                # Draw circle and ID with better visibility
                cv2.circle(frame, (center_x, center_y), 12, (0, 255, 0), 3)
                cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)
                
                # Add background rectangle for better text visibility
                text = f"P{track_id}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, 
                            (center_x - text_size[0]//2 - 5, center_y - 25),
                            (center_x + text_size[0]//2 + 5, center_y - 10),
                            (0, 0, 0), -1)
                cv2.putText(frame, text, 
                          (center_x - text_size[0]//2, center_y - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0:  # Every 30 frames
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end='\r')
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Final tracked video saved to {output_path}")

if __name__ == "__main__":
    main()