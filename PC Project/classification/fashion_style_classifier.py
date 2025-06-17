import torch
from PIL import Image
import numpy as np

class FashionStyleClassifier:
    def __init__(self, model_path="fashion_style_model.pt"):
        self.model = torch.load(model_path)
        self.model.eval()

    def classify(self, image: np.ndarray) -> str:
        img_pil = Image.fromarray(image).resize((224, 224))
        img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        style_mapping = {0: "캐주얼", 1: "포멀", 2: "스포티", 3: "빈티지", 4: "스트릿"}
        return style_mapping[predicted.item()]
