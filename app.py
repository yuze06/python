"""
Streamlit 網頁版本 - 衣櫃穿搭推薦系統
可以直接在瀏覽器上運行，無需 Colab

安裝: pip install streamlit
運行: streamlit run app.py
"""

import streamlit as st
import os
import json
import uuid
import math
import requests
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ===== 頁面配置 =====
st.set_page_config(
    page_title="👔 衣櫃穿搭推薦系統",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS 美化 =====
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .stButton > button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    h1 {
        color: #FF69B4;
        text-align: center;
    }
    h2 {
        color: #4169E1;
    }
</style>
""", unsafe_allow_html=True)

# ===== 會話狀態初始化 =====
if 'closet' not in st.session_state:
    st.session_state.closet = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ===== 配置 =====
BASE_DIR = "./closet_project"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CLOSET_FILE = os.path.join(BASE_DIR, "closet.json")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ===== 常數 =====
CLOTHING_LABELS = [
    "a photo of a t-shirt", "a photo of a shirt", "a photo of a blouse",
    "a photo of a polo shirt", "a photo of a hoodie", "a photo of a sweater",
    "a photo of a jacket", "a photo of a coat", "a photo of a blazer",
    "a photo of a dress", "a photo of a skirt", "a photo of pants",
    "a photo of jeans", "a photo of shorts", "a photo of sneakers",
    "a photo of shoes", "a photo of sandals", "a photo of boots",
    "a photo of a bag", "a photo of a hat"
]

COLOR_PALETTE = {
    "black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128),
    "red": (200, 50, 50), "blue": (50, 80, 200), "green": (50, 140, 70),
    "yellow": (220, 200, 60), "brown": (120, 80, 50), "pink": (230, 150, 180),
    "purple": (140, 90, 170), "orange": (220, 120, 50), "beige": (220, 200, 170),
}

CATEGORY_MAP = {
    "t-shirt": "上衣", "shirt": "上衣", "blouse": "上衣", "polo": "上衣",
    "hoodie": "上衣", "sweater": "上衣", "dress": "連衣裙",
    "jacket": "外套", "coat": "外套", "blazer": "外套",
    "pants": "褲子", "jeans": "牛仔褲", "shorts": "短褲", "skirt": "裙子",
    "sneakers": "運動鞋", "shoes": "鞋子", "sandals": "涼鞋", "boots": "靴子",
    "bag": "包包", "hat": "帽子"
}

# ===== 數據類 =====
@dataclass
class ClothingItem:
    id: str
    name: str
    category: str
    color: str
    material: str
    thickness: str
    style: str
    image_path: Optional[str] = None
    added_date: Optional[str] = None

# ===== 衣櫃管理函數 =====

def load_closet() -> List[ClothingItem]:
    """載入衣櫃"""
    if not os.path.exists(CLOSET_FILE):
        return []
    try:
        with open(CLOSET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ClothingItem(**item) for item in data]
    except:
        return []

def save_closet(clothes: List[ClothingItem]) -> None:
    """保存衣櫃"""
    try:
        with open(CLOSET_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in clothes], f, 
                     ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存失敗: {e}")

def add_clothing(name: str, category: str, color: str, material: str, 
                thickness: str, style: str, image_path: str = None) -> ClothingItem:
    """添加衣服"""
    item = ClothingItem(
        id=str(uuid.uuid4())[:8],
        name=name,
        category=category,
        color=color,
        material=material,
        thickness=thickness,
        style=style,
        image_path=image_path,
        added_date=datetime.now().isoformat()
    )
    clothes = load_closet()
    clothes.append(item)
    save_closet(clothes)
    return item

def delete_clothing(item_id: str) -> bool:
    """刪除衣服"""
    clothes = load_closet()
    original_len = len(clothes)
    clothes = [c for c in clothes if c.id != item_id]
    if len(clothes) < original_len:
        save_closet(clothes)
        return True
    return False

# ===== 圖像分析函數 =====

@st.cache_resource
def load_clip_model():
    """載入 CLIP 模型 (緩存)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return processor, model, device

def predict_clothing_label(image_path: str) -> Tuple[str, float]:
    """預測衣服標籤"""
    try:
        processor, model, device = load_clip_model()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=CLOTHING_LABELS, images=image, 
                         return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        best_idx = int(probs.argmax())
        return CLOTHING_LABELS[best_idx], float(probs[best_idx])
    except Exception as e:
        st.error(f"預測失敗: {e}")
        return "unknown", 0.0

def predict_color(image_path: str) -> Tuple[str, Tuple[int, int, int]]:
    """預測主色調"""
    try:
        img = Image.open(image_path).convert("RGB").resize((100, 100))
        pixels = list(img.getdata())
        
        filtered = [p for p in pixels if not (p[0] > 245 and p[1] > 245 and p[2] > 245)]
        if not filtered:
            filtered = pixels
        
        avg = tuple(sum(c[i] for c in filtered) // len(filtered) for i in range(3))
        
        def dist(c1, c2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
        
        best_color = min(COLOR_PALETTE.items(), 
                        key=lambda x: dist(avg, x[1]))[0]
        
        return best_color, avg
    except:
        return "gray", (128, 128, 128)

def map_label_to_category(label: str) -> str:
    """將標籤對應到類別"""
    text = label.lower()
    for keyword, cat_cn in CATEGORY_MAP.items():
        if keyword in text:
            # 反轉對應
            for k, v in CATEGORY_MAP.items():
                if v == cat_cn and keyword in k:
                    return keyword.replace("s", "").replace("es", "")
    return "top"

def infer_material_thickness(category: str, label: str) -> Tuple[str, str]:
    """推斷材質和厚薄"""
    label_lower = label.lower()
    material, thickness = "棉", "中等"
    
    if "outer" in category or any(x in label_lower for x in ["jacket", "coat"]):
        if any(x in label_lower for x in ["coat", "blazer"]):
            thickness, material = "厚", "聚酯纖維"
        else:
            thickness, material = "中等", "合成纖維"
    elif "top" in category:
        if any(x in label_lower for x in ["t-shirt", "shirt", "blouse"]):
            thickness, material = "薄", "棉"
        elif any(x in label_lower for x in ["hoodie", "sweater"]):
            thickness, material = "中等", "針織"
    elif "bottom" in category:
        if "jeans" in label_lower:
            thickness, material = "中等", "牛仔布"
        else:
            thickness, material = "薄", "棉"
    
    return material, thickness

# ===== 主頁面 =====

st.title("👔 智慧衣櫃穿搭推薦系統")

# 側邊欄導航
with st.sidebar:
    st.title("🎯 菜單")
    page = st.radio("選擇功能", 
                   ["📊 首頁", "👕 添加衣服", "👀 查看衣櫃", 
                    "🎨 穿搭推薦", "🔍 搜尋衣服"])

# ===== ��頁 =====
if page == "📊 首頁":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    clothes = load_closet()
    
    with col1:
        st.metric("📦 衣服總數", len(clothes))
    
    with col2:
        categories = set(c.category for c in clothes)
        st.metric("🏷️ 衣服類別", len(categories))
    
    with col3:
        colors = set(c.color for c in clothes)
        st.metric("🎨 顏色數量", len(colors))
    
    st.markdown("---")
    
    st.subheader("📈 統計分析")
    
    if clothes:
        col1, col2 = st.columns(2)
        
        with col1:
            # 按類別統計
            cat_counts = {}
            for c in clothes:
                cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
            
            st.write("**按類別分布:**")
            for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  • {cat}: {count} 件")
        
        with col2:
            # 按顏色統計
            color_counts = {}
            for c in clothes:
                color_counts[c.color] = color_counts.get(c.color, 0) + 1
            
            st.write("**按顏色分布:**")
            for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                st.write(f"  • {color}: {count} 件")

# ===== 添加衣服 =====
elif page == "👕 添加衣服":
    st.subheader("➕ 添加新衣服")
    
    st.info("📝 方式 1: 自動分析 (上傳圖片)")
    
    uploaded_file = st.file_uploader("上傳衣服圖片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 保存圖片
        image_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 顯示圖片
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="上傳的圖片", use_column_width=True)
        
        with col2:
            st.write("🔍 正在分析...")
            
            with st.spinner("分析中..."):
                # 預測
                label, confidence = predict_clothing_label(image_path)
                color, _ = predict_color(image_path)
                material, thickness = infer_material_thickness("top", label)
                
                st.success("✅ 分析完成!")
                
                st.write(f"**檢測到的衣服類型:**")
                st.write(f"  • 標籤: {label}")
                st.write(f"  • 信心度: {confidence:.1%}")
                st.write(f"  • 顏色: {color}")
                st.write(f"  • 材質: {material}")
                st.write(f"  • 厚薄: {thickness}")
        
        st.divider()
        
        st.write("📝 編輯詳情:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("衣服名稱", value=uploaded_file.name.split('.')[0])
            category_options = list(set(CATEGORY_MAP.values()))
            category = st.selectbox("類別", category_options, 
                                   index=0 if "上衣" in category_options else 0)
            color_selected = st.selectbox("顏色", list(COLOR_PALETTE.keys()), 
                                         index=list(COLOR_PALETTE.keys()).index(color) 
                                         if color in COLOR_PALETTE else 0)
        
        with col2:
            material_options = ["棉", "聚酯纖維", "合成纖維", "牛仔布", "針織", "羊毛"]
            material_selected = st.selectbox("材質", material_options,
                                            index=material_options.index(material) 
                                            if material in material_options else 0)
            thickness_options = ["薄", "中等", "厚"]
            thickness_selected = st.selectbox("厚薄", thickness_options,
                                             index=thickness_options.index(thickness)
                                             if thickness in thickness_options else 0)
            style_options = ["休閒", "上班", "運動", "正式"]
            style = st.selectbox("風格", style_options)
        
        if st.button("✅ 確認添加", use_container_width=True):
            add_clothing(
                name=name,
                category=category,
                color=color_selected,
                material=material_selected,
                thickness=thickness_selected,
                style=style,
                image_path=image_path
            )
            st.success(f"✅ 已添加: {name}")
            st.balloons()
    
    st.divider()
    st.info("📝 方式 2: 手動輸入")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("衣服名稱 (手動)", key="manual_name")
        category = st.selectbox("類別 (手動)", list(set(CATEGORY_MAP.values())), key="manual_cat")
    
    with col2:
        color = st.selectbox("顏色 (手動)", list(COLOR_PALETTE.keys()), key="manual_color")
        style = st.selectbox("風格 (手動)", ["休閒", "上班", "運動"], key="manual_style")
    
    if st.button("✅ 手動添加", use_container_width=True):
        add_clothing(name, category, color, "棉", "中等", style)
        st.success(f"✅ 已添加: {name}")
        st.balloons()

# ===== 查看衣櫃 =====
elif page == "👀 查看衣櫃":
    st.subheader("📦 我的衣櫃")
    
    clothes = load_closet()
    
    if not clothes:
        st.info("衣櫃還是空的，快去添加衣服吧！")
    else:
        # 篩選選項
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_category = st.selectbox("按類別篩選", 
                                          ["全部"] + list(set(c.category for c in clothes)))
        with col2:
            filter_color = st.selectbox("按顏色篩選",
                                       ["全部"] + list(set(c.color for c in clothes)))
        with col3:
            filter_style = st.selectbox("按風格篩選",
                                       ["全部"] + list(set(c.style for c in clothes)))
        
        # 應用篩選
        filtered = clothes
        if filter_category != "全部":
            filtered = [c for c in filtered if c.category == filter_category]
        if filter_color != "全部":
            filtered = [c for c in filtered if c.color == filter_color]
        if filter_style != "全部":
            filtered = [c for c in filtered if c.style == filter_style]
        
        st.write(f"**找到 {len(filtered)} 件衣服**")
        
        # 顯示表格
        df = pd.DataFrame([{
            "ID": c.id,
            "名稱": c.name,
            "類別": c.category,
            "顏色": c.color,
            "厚薄": c.thickness,
            "風格": c.style,
            "材質": c.material,
            "添加日期": c.added_date[:10] if c.added_date else "-"
        } for c in filtered])
        
        st.dataframe(df, use_container_width=True)
        
        st.divider()
        
        st.write("🗑️ **刪除衣服**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_name = st.selectbox("選擇要刪除的衣服", 
                                        [f"{c.name} ({c.color})" for c in filtered],
                                        key="delete_select")
        
        with col2:
            if st.button("🗑️ 刪除", use_container_width=True):
                for c in filtered:
                    if f"{c.name} ({c.color})" == selected_name:
                        delete_clothing(c.id)
                        st.success("✅ 已刪除")
                        st.rerun()

# ===== 穿搭推薦 =====
elif page == "🎨 穿搭推薦":
    st.subheader("👗 今日穿搭推薦")
    
    clothes = load_closet()
    
    if not clothes:
        st.warning("衣櫃為空，無法推薦穿搭")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            style = st.selectbox("選擇風格", ["休閒", "上班", "運動"])
        
        with col2:
            num_outfits = st.slider("推薦套數", 1, 5, 3)
        
        if st.button("🎲 生成穿搭", use_container_width=True):
            # 按風格篩選
            style_clothes = [c for c in clothes if c.style == style]
            
            if not style_clothes:
                st.warning(f"沒有 {style} 風格的衣服")
            else:
                # 按類別分組
                by_cat = {}
                for c in style_clothes:
                    if c.category not in by_cat:
                        by_cat[c.category] = []
                    by_cat[c.category].append(c)
                
                # 生成推薦
                for i in range(num_outfits):
                    st.write(f"### 👗 穿搭方案 {i+1}")
                    
                    outfit = []
                    
                    # 上衣
                    if "上衣" in by_cat:
                        outfit.append(by_cat["上衣"][i % len(by_cat["上衣"])])
                    
                    # 褲子/裙子
                    bottom_cats = [cat for cat in by_cat.keys() if "褲" in cat or "裙" in cat]
                    if bottom_cats:
                        outfit.append(by_cat[bottom_cats[0]][i % len(by_cat[bottom_cats[0]])])
                    
                    # 鞋子
                    shoe_cats = [cat for cat in by_cat.keys() if "鞋" in cat]
                    if shoe_cats:
                        outfit.append(by_cat[shoe_cats[0]][i % len(by_cat[shoe_cats[0]])])
                    
                    # 顯示穿搭
                    cols = st.columns(len(outfit))
                    for j, item in enumerate(outfit):
                        with cols[j]:
                            st.write(f"**{item.name}**")
                            st.write(f"顏色: {item.color}")
                            st.write(f"厚薄: {item.thickness}")
                    
                    st.divider()

# ===== 搜尋衣服 =====
elif page == "🔍 搜尋衣服":
    st.subheader("🔍 搜尋衣服")
    
    search_query = st.text_input("輸入搜尋關鍵詞 (名稱、顏色等)")
    
    clothes = load_closet()
    
    if search_query:
        query_lower = search_query.lower()
        results = [c for c in clothes if query_lower in c.name.lower() or 
                  query_lower in c.color.lower() or
                  query_lower in c.category.lower()]
        
        if results:
            st.success(f"✅ 找到 {len(results)} 件")
            
            df = pd.DataFrame([{
                "名稱": c.name,
                "類別": c.category,
                "顏色": c.color,
                "厚薄": c.thickness,
                "風格": c.style
            } for c in results])
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"❌ 沒有找到包含 '{search_query}' 的衣服")
    else:
        st.info("輸入搜尋關鍵詞開始搜尋")

# ===== 底部信息 =====
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>👗 智慧衣櫃穿搭推薦系統 v2.0</p>
    <p>Made with ❤️ by AI Fashion Assistant</p>
</div>
""", unsafe_allow_html=True)
