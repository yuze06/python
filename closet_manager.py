"""
衣櫃管理模組 - JSON 數據持久化
"""
import os
import json
import uuid
from typing import List, Dict, Optional
from dataclasses import asdict, dataclass
from datetime import datetime

from config import config


@dataclass
class ClothingItem:
    """衣服項目數據類"""
    id: str
    name: str
    category: str  # top / bottom / outer / shoes / accessory
    color: str
    material: str
    thickness: str  # thin / medium / thick
    style: str  # student / casual / office / formal / sport
    image_path: Optional[str] = None
    added_date: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.added_date:
            self.added_date = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []


class ClosetManager:
    """衣櫃管理器"""
    
    def __init__(self, closet_file: Optional[str] = None):
        self.closet_file = closet_file or config.CLOSET_FILE
        self._ensure_directory()
    
    def _ensure_directory(self) -> None:
        """確保目錄存在"""
        os.makedirs(os.path.dirname(self.closet_file), exist_ok=True)
    
    def load(self) -> List[ClothingItem]:
        """載入衣櫃"""
        if not os.path.exists(self.closet_file):
            return []
        
        try:
            with open(self.closet_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [ClothingItem(**item) for item in data]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"載入衣櫃失敗: {e}")
            return []
    
    def save(self, items: List[ClothingItem]) -> None:
        """保存衣櫃"""
        try:
            with open(self.closet_file, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(item) for item in items],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except IOError as e:
            print(f"保存衣櫃失敗: {e}")
    
    def add_item(
        self,
        name: str,
        category: str,
        color: str,
        material: str,
        thickness: str,
        style: str,
        image_path: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ClothingItem:
        """添加衣服項目"""
        item = ClothingItem(
            id=str(uuid.uuid4()),
            name=name,
            category=category,
            color=color,
            material=material,
            thickness=thickness,
            style=style,
            image_path=image_path,
            tags=tags or []
        )
        
        clothes = self.load()
        clothes.append(item)
        self.save(clothes)
        
        return item
    
    def remove_item(self, item_id: str) -> bool:
        """移除衣服項目"""
        clothes = self.load()
        original_len = len(clothes)
        clothes = [item for item in clothes if item.id != item_id]
        
        if len(clothes) < original_len:
            self.save(clothes)
            return True
        return False
    
    def update_item(self, item_id: str, **kwargs) -> Optional[ClothingItem]:
        """更新衣服項目"""
        clothes = self.load()
        
        for item in clothes:
            if item.id == item_id:
                for key, value in kwargs.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                self.save(clothes)
                return item
        
        return None
    
    def get_by_category(self, category: str) -> List[ClothingItem]:
        """按類別篩選衣服"""
        return [item for item in self.load() if item.category == category]
    
    def get_by_style(self, style: str) -> List[ClothingItem]:
        """按風格篩選衣服"""
        return [item for item in self.load() if item.style == style]
    
    def get_by_color(self, color: str) -> List[ClothingItem]:
        """按顏色篩選衣服"""
        return [item for item in self.load() if item.color.lower() == color.lower()]
    
    def search(self, query: str) -> List[ClothingItem]:
        """搜尋衣服"""
        query_lower = query.lower()
        return [
            item for item in self.load()
            if query_lower in item.name.lower() or
               query_lower in str(item.tags).lower()
        ]
    
    def get_statistics(self) -> Dict:
        """獲取衣櫃統計"""
        clothes = self.load()
        
        return {
            "total": len(clothes),
            "by_category": self._count_by_field(clothes, "category"),
            "by_color": self._count_by_field(clothes, "color"),
            "by_style": self._count_by_field(clothes, "style"),
            "by_thickness": self._count_by_field(clothes, "thickness"),
        }
    
    @staticmethod
    def _count_by_field(items: List[ClothingItem], field: str) -> Dict[str, int]:
        """統計某個欄位的分布"""
        counts = {}
        for item in items:
            value = getattr(item, field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def clear(self) -> None:
        """清空衣櫃"""
        self.save([])
        print("衣櫃已清空")
    
    def export_json(self, export_path: str) -> None:
        """導出為 JSON"""
        clothes = self.load()
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(item) for item in clothes],
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"已導出到: {export_path}")
    
    def import_json(self, import_path: str) -> int:
        """從 JSON 導入"""
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = [ClothingItem(**item) for item in data]
                self.save(items)
                print(f"已導入 {len(items)} 件衣服")
                return len(items)
        except (IOError, json.JSONDecodeError) as e:
            print(f"導入失敗: {e}")
            return 0
