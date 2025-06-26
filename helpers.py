import cv2
import numpy as np

def crop_bbox(image, bbox, padding=0):
    """
    Crop image using bounding box coordinates with optional padding.
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        padding: Additional padding around the bbox
        
    Returns:
        Cropped image or empty array if invalid
    """
    if image is None or len(image.shape) < 2:
        return np.array([])
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Validate coordinates
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    
    # Crop and return
    cropped = image[y1:y2, x1:x2]
    
    # Check if crop is valid
    if cropped.size == 0:
        return np.array([])
    
    return cropped

def draw_boxes(frame, tracked_objects, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and IDs for tracked players.
    
    Args:
        frame: Input frame
        tracked_objects: List of tracked objects from Norfair
        color: RGB color tuple for drawing
        thickness: Line thickness
        
    Returns:
        Frame with drawn boxes and IDs
    """
    for obj in tracked_objects:
        if len(obj.estimate) == 0:
            continue
            
        # Get center point
        center_x, center_y = map(int, obj.estimate[0])
        track_id = obj.id
        
        # Get bounding box if available
        if obj.last_detection and obj.last_detection.data:
            bbox = obj.last_detection.data.get('bbox')
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Draw ID label with background
                label = f"Player {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(frame, 
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0] + 10, y1),
                             color, -1)
                
                # Text
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Fallback: just draw center point
            cv2.circle(frame, (center_x, center_y), 8, color, -1)
            cv2.putText(frame, f"P{track_id}", 
                       (center_x - 10, center_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def draw_tracking_info(frame, tracked_objects, show_trails=False):
    """
    Draw comprehensive tracking information on frame.
    
    Args:
        frame: Input frame
        tracked_objects: List of tracked objects
        show_trails: Whether to show tracking trails
        
    Returns:
        Frame with tracking visualization
    """
    # Color palette for different players
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
    
    for i, obj in enumerate(tracked_objects):
        if len(obj.estimate) == 0:
            continue
            
        # Use different color for each track
        color = colors[obj.id % len(colors)]
        
        center_x, center_y = map(int, obj.estimate[0])
        track_id = obj.id
        
        # Draw main circle
        cv2.circle(frame, (center_x, center_y), 12, color, 3)
        cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)
        
        # Draw ID with background
        text = f"P{track_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background rectangle
        cv2.rectangle(frame,
                     (center_x - text_size[0]//2 - 5, center_y - 30),
                     (center_x + text_size[0]//2 + 5, center_y - 10),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, text,
                   (center_x - text_size[0]//2, center_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show confidence if available
        if obj.last_detection and obj.last_detection.data:
            conf = obj.last_detection.data.get('confidence', 0)
            conf_text = f"{conf:.2f}"
            cv2.putText(frame, conf_text,
                       (center_x - 15, center_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def validate_bbox(bbox, frame_shape):
    """
    Validate and clip bounding box to frame boundaries.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_shape: Frame shape (height, width, channels)
        
    Returns:
        Clipped bounding box or None if invalid
    """
    if bbox is None:
        return None
    
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Clip to frame boundaries
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    
    # Check if bbox is valid
    if x2 <= x1 or y2 <= y1:
        return None
    
    return (x1, y1, x2, y2)