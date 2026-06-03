"""
配置管理模組
"""
import os
from dataclasses import dataclass
from typing import Dict, Tuple

# ============================================================
# 配置常數
# ============================================================

@dataclass
class Config:
    """主配置類"""
    # 中央氣象署 API Key
    CWA_API_KEY: str = "你的中央氣象署API授權碼"
    
    # 城市
    CITY_NAME: str = "臺中市"
    
    # 是否掛載 Google Drive
    USE_GOOGLE_DRIVE: bool = False
    
    # 是否啟用 LINE 推播
    ENABLE_LINE_PUSH: bool = False
    LINE_CHANNEL_ACCESS_TOKEN: str = ""
    LINE_USER_ID: str = ""
    
    # 默認穿搭風格
    DEFAULT_STYLE_MODE: str = "casual"  # student / casual / office
    
    # 基本路徑
    @property
    def BASE_DIR(self) -> str:
        if self.USE_GOOGLE_DRIVE:
            return "/content/drive/MyDrive/weather_outfit_project"
        return "/content/weather_outfit_project"
    
    @property
    def IMAGE_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "closet_images")
    
    @property
    def CLOSET_FILE(self) -> str:
        return os.path.join(self.BASE_DIR, "closet_data.json")


# 顏色調色盤
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "red": (200, 50, 50),
    "blue": (50, 80, 200),
    "green": (50, 140, 70),
    "yellow": (220, 200, 60),
    "brown": (120, 80, 50),
    "pink": (230, 150, 180),
    "purple": (140, 90, 170),
    "orange": (220, 120, 50),
    "beige": (220, 200, 170),
    "lightblue": (150, 190, 230)
}

# CLIP 模型的衣服標籤
CLOTHING_LABELS = [
    "a photo of a t-shirt",
    "a photo of a shirt",
    "a photo of a blouse",
    "a photo of a polo shirt",
    "a photo of a hoodie",
    "a photo of a sweater",
    "a photo of a jacket",
    "a photo of a coat",
    "a photo of a blazer",
    "a photo of a dress",
    "a photo of a skirt",
    "a photo of pants",
    "a photo of jeans",
    "a photo of shorts",
    "a photo of sneakers",
    "a photo of shoes",
    "a photo of sandals",
    "a photo of boots",
    "a photo of a bag",
    "a photo of a hat"
]

# 衣服類別對應規則
CATEGORY_MAPPING = {
    "top": ["t-shirt", "shirt", "blouse", "polo", "hoodie", "sweater", "dress"],
    "outer": ["jacket", "coat", "blazer"],
    "bottom": ["pants", "jeans", "shorts", "skirt"],
    "shoes": ["sneakers", "shoes", "sandals", "boots"],
    "accessory": ["bag", "hat"]
}

# 材質推斷規則
MATERIAL_RULES = {
    "outer": {
        "thick": ["coat", "blazer"],
        "medium": ["jacket"],
        "material": "synthetic"
    },
    "top": {
        "thin": ["t-shirt", "shirt", "blouse", "polo"],
        "medium": ["hoodie", "sweater"],
    },
    "bottom": {
        "thin": ["shorts", "skirt"],
        "medium": ["jeans", "pants"],
    }
}

# 預設配置實例
config = Config()
