"""
圖像處理模組 - 衣服識別、顏色檢測
"""
import os
import math
from typing import Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel

from config import COLOR_PALETTE, CLOTHING_LABELS, CATEGORY_MAPPING


class ClothingDetector:
    """衣服檢測器 - 使用 CLIP 模型"""
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ClothingDetector._model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用設備: {self.device}")
            
            model_name = "openai/clip-vit-base-patch32"
            ClothingDetector._processor = CLIPProcessor.from_pretrained(model_name)
            ClothingDetector._model = CLIPModel.from_pretrained(model_name).to(self.device)
            print("CLIP 模型載入成功")
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value
    
    def predict_label(self, image_path: str) -> Tuple[str, float]:
        """
        預測衣服標籤
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            (最佳標籤, 信心度)
        """
        image = Image.open(image_path).convert("RGB")
        
        inputs = self._processor(
            text=CLOTHING_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        best_idx = int(probs.argmax())
        best_label = CLOTHING_LABELS[best_idx]
        best_score = float(probs[best_idx])
        
        return best_label, best_score
    
    @staticmethod
    def map_label_to_category(label: str) -> str:
        """將標籤對應到衣服類別"""
        text = label.lower()
        
        for category, keywords in CATEGORY_MAPPING.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return "top"  # 預設值


class ColorDetector:
    """顏色檢測器"""
    
    @staticmethod
    def rgb_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        """計算 RGB 顏色距離"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    @classmethod
    def predict_dominant_color(
        cls,
        image_path: str,
        resize: Tuple[int, int] = (100, 100)
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        預測主色調
        
        Args:
            image_path: 圖片路徑
            resize: 調整大小
            
        Returns:
            (顏色名稱, RGB 值)
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize(resize)
        
        pixels = list(img.getdata())
        
        # 過濾太白或太黑的像素
        filtered = [
            p for p in pixels
            if not (p[0] > 245 and p[1] > 245 and p[2] > 245)
        ]
        
        if not filtered:
            filtered = pixels
        
        # 計算平均顏色
        avg = tuple(
            sum(c[i] for c in filtered) // len(filtered) for i in range(3)
        )
        
        # 找最接近的顏色
        best_color = None
        best_dist = float("inf")
        
        for color_name, ref_rgb in COLOR_PALETTE.items():
            d = cls.rgb_distance(avg, ref_rgb)
            if d < best_dist:
                best_dist = d
                best_color = color_name
        
        return best_color, avg


class ImageViewer:
    """圖像檢視工具"""
    
    @staticmethod
    def show_image(path: str, title: Optional[str] = None) -> None:
        """顯示圖片"""
        if not path or not os.path.exists(path):
            print(f"找不到圖片: {path}")
            return
        
        img = Image.open(path)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()
    
    @staticmethod
    def show_clothing_batch(image_paths: list, titles: Optional[list] = None) -> None:
        """批量顯示衣服圖片"""
        n = len(image_paths)
        if n == 0:
            return
        
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for idx, path in enumerate(image_paths):
            if os.path.exists(path):
                img = Image.open(path)
                axes[idx].imshow(img)
                axes[idx].axis("off")
                if titles and idx < len(titles):
                    axes[idx].set_title(titles[idx])
        
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.show()
