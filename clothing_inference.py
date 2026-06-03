"""
衣服推斷模組 - 材質、厚薄、風格推斷
"""
from typing import Tuple
from config import CATEGORY_MAPPING, MATERIAL_RULES


class ClothingInference:
    """衣服屬性推斷"""
    
    # 材質對應
    MATERIAL_MAP = {
        "thin": {
            "top": "cotton",
            "bottom": "cotton",
            "outer": "silk",
        },
        "medium": {
            "top": "knit",
            "bottom": "polyester",
            "outer": "synthetic",
        },
        "thick": {
            "top": "wool",
            "bottom": "denim",
            "outer": "polyester",
        }
    }
    
    @staticmethod
    def infer_material_and_thickness(
        category: str,
        label_text: str
    ) -> Tuple[str, str]:
        """
        推斷材質和厚薄
        
        Args:
            category: 衣服類別
            label_text: CLIP 預測的標籤文本
            
        Returns:
            (材質, 厚薄)
        """
        label_lower = label_text.lower()
        material = "cotton"
        thickness = "medium"
        
        # 外套
        if category == "outer":
            if any(x in label_lower for x in ["coat", "blazer"]):
                thickness = "thick"
                material = "polyester"
            elif "jacket" in label_lower:
                thickness = "medium"
                material = "synthetic"
        
        # 上衣
        elif category == "top":
            if any(x in label_lower for x in ["t-shirt", "shirt", "blouse", "polo"]):
                thickness = "thin"
                material = "cotton"
            elif any(x in label_lower for x in ["hoodie", "sweater"]):
                thickness = "medium"
                material = "knit"
        
        # 下裝
        elif category == "bottom":
            if "jeans" in label_lower:
                thickness = "medium"
                material = "denim"
            elif any(x in label_lower for x in ["shorts", "skirt"]):
                thickness = "thin"
                material = "cotton"
            else:
                thickness = "medium"
                material = "polyester"
        
        # 鞋子
        elif category == "shoes":
            thickness = "medium"
            material = "synthetic"
        
        return material, thickness
    
    @staticmethod
    def infer_style(
        category: str,
        label_text: str,
        color: str
    ) -> str:
        """
        推斷穿搭風格
        
        Args:
            category: 衣服類別
            label_text: CLIP 預測的標籤文本
            color: 顏色
            
        Returns:
            風格 (student / casual / office / formal / sport)
        """
        label_lower = label_text.lower()
        
        # 正式風格
        if any(x in label_lower for x in ["blazer", "formal", "suit"]):
            return "office"
        
        # 運動風格
        if any(x in label_lower for x in ["sneakers", "sports", "gym"]):
            return "sport"
        
        # 根據顏色推斷
        dark_colors = ["black", "gray", "navy"]
        bright_colors = ["yellow", "orange", "red", "pink"]
        
        if color.lower() in dark_colors:
            return "office"
        elif color.lower() in bright_colors:
            return "casual"
        
        # 預設
        return "casual"
    
    @staticmethod
    def get_recommended_thickness_for_weather(
        temp: float,
        humidity: float,
        weather_type: str
    ) -> str:
        """
        根據天氣推薦厚薄
        
        Args:
            temp: 溫度 (攝氏度)
            humidity: 濕度 (%)
            weather_type: 天氣類型
            
        Returns:
            推薦厚薄 (thin / medium / thick)
        """
        # 下雨 / 雪天，建議厚一點
        if any(x in weather_type for x in ["rain", "snow", "storm"]):
            if temp < 10:
                return "thick"
            elif temp < 18:
                return "medium"
            else:
                return "thin"
        
        # 根據溫度
        if temp < 5:
            return "thick"
        elif temp < 15:
            return "medium"
        elif temp < 25:
            return "thin"
        else:
            return "thin"
    
    @staticmethod
    def get_color_compatibility(
        color1: str,
        color2: str
    ) -> float:
        """
        計算兩種顏色的搭配度 (0-1)
        
        Args:
            color1: 顏色1
            color2: 顏色2
            
        Returns:
            搭配度 (0-1)
        """
        # 簡化的色彩搭配規則
        compatible_pairs = {
            ("black", "white"): 0.95,
            ("black", "gray"): 0.90,
            ("white", "gray"): 0.85,
            ("blue", "white"): 0.90,
            ("blue", "gray"): 0.85,
            ("red", "black"): 0.85,
            ("red", "white"): 0.80,
            ("brown", "beige"): 0.90,
            ("brown", "white"): 0.85,
        }
        
        # 檢查對稱對
        if (color1, color2) in compatible_pairs:
            return compatible_pairs[(color1, color2)]
        elif (color2, color1) in compatible_pairs:
            return compatible_pairs[(color2, color1)]
        
        # 相同顏色
        if color1.lower() == color2.lower():
            return 0.70
        
        # 預設相容度
        return 0.60
