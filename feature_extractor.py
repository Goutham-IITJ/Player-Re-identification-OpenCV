import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

class FeatureExtractor:
    def __init__(self, device='cpu'):
        """
        Initialize ResNet50 feature extractor for person re-identification.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Use ResNet50 with ImageNet pretrained weights, no classification head
        self.model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        
        # Standard transforms for person re-ID
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Standard re-ID input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                        std=[0.229, 0.224, 0.225])
        ])
        
        print(f"🧠 Feature extractor initialized on {device}")

    def extract(self, image):
        """
        Extract normalized feature vector from cropped player image.
        
        Args:
            image: Cropped player image (numpy array, BGR format)
            
        Returns:
            Normalized feature vector (numpy array) or None if extraction fails
        """
        if image is None or image.size == 0:
            print("⚠️ Empty image provided to feature extractor")
            return None

        try:
            # Convert BGR to RGB for PIL
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print("⚠️ Invalid image format")
                return None
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # Convert to numpy and normalize
            features = features.squeeze().cpu().numpy()
            
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            else:
                print("⚠️ Zero norm feature vector")
                return None
                
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return None

    def extract_batch(self, images):
        """
        Extract features from multiple images in batch.
        
        Args:
            images: List of cropped player images
            
        Returns:
            List of normalized feature vectors
        """
        features_list = []
        
        for img in images:
            feature = self.extract(img)
            features_list.append(feature)
            
        return features_list

    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            feat1, feat2: Feature vectors
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if feat1 is None or feat2 is None:
            return 0.0
            
        try:
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            return max(0.0, similarity)  # Clamp to [0, 1]
        except:
            return 0.0