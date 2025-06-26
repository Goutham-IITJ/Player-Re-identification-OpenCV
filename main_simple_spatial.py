import cv2
import os
from yolo_detector import YOLODetector
from feature_extractor import FeatureExtractor
from simple_spatial_tracker import SimpleSpatialTracker
from helpers import crop_bbox

def main():
    video_path = "15sec_input_720p.mp4"
    output_path = "outputs/tracked_players_spatial.mp4"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize components
    detector = YOLODetector("best.pt", device="cpu")
    extractor = FeatureExtractor(device="cpu")
    tracker = SimpleSpatialTracker(max_distance=80, max_disappeared=45)  # Simple spatial tracker
    
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
    print(f"🎥 Processing {total_frames} frames with Simple Spatial Tracker...")
    
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
            
            # Filter out very small detections
            width_bbox = x2 - x1
            height_bbox = y2 - y1
            if width_bbox < 25 or height_bbox < 40:  # Even smaller minimum for better detection
                continue
                
            # Ensure bbox is within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Still extract features for potential future use
            cropped = crop_bbox(frame, (x1, y1, x2, y2))
            if cropped.size > 0:
                feature = extractor.extract(cropped)
                det["bbox"] = (x1, y1, x2, y2)
                det["feature"] = feature
                valid_detections.append(det)
        
        # Update tracker
        active_tracks = tracker.update(valid_detections)
        
        # Draw tracking results
        for track in active_tracks:
            center_x, center_y = map(int, track['center'])
            track_id = track['id']
            
            # Ensure center is within frame bounds
            center_x = max(15, min(width - 15, center_x))
            center_y = max(25, min(height - 15, center_y))
            
            # Draw with different colors for different IDs
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue  
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (255, 165, 0),  # Orange
            ]
            
            color = colors[track_id % len(colors)]
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), 12, color, 3)
            cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)
            
            # Add background rectangle for better text visibility
            text = f"P{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, 
                        (center_x - text_size[0]//2 - 5, center_y - 30),
                        (center_x + text_size[0]//2 + 5, center_y - 10),
                        (0, 0, 0), -1)
            cv2.putText(frame, text, 
                      (center_x - text_size[0]//2, center_y - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        out.write(frame)
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) - Active tracks: {len(active_tracks)}", end='\r')
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Final tracked video saved to {output_path}")

if __name__ == "__main__":
    main()