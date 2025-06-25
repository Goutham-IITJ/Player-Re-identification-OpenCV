import torch
import torchvision.transforms as T
import timm
import numpy as np

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            feature = self.model(img_tensor).squeeze().cpu().numpy()
        return feature / np.linalg.norm(feature)
