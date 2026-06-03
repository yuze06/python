"""
Streamlit 網頁版本 - 簡化版 (不需要 torch)
直接可在 Streamlit Cloud 上運行

安裝: pip install streamlit requests pandas pillow matplotlib
運行: streamlit run app_simple.py
"""

import streamlit as st
import os
import json
import uuid
import math
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from PIL import Image

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
        background-color: #FF69B4;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #FF1493;
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

# ===== 配置 =====
BASE_DIR = "./closet_project"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CLOSET_FILE = os.path.join(BASE_DIR, "closet.json")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ===== 常數 =====
COLOR_PALETTE = {
    "black": "黑色", "white": "白色", "gray": "灰色",
    "red": "紅色", "blue": "藍色", "green": "綠色",
    "yellow": "黃色", "brown": "棕色", "pink": "粉紅色",
    "purple": "紫色", "orange": "橙色", "beige": "米色",
}

CATEGORIES = ["上衣", "褲子", "裙子", "外套", "鞋子", "包包", "帽子"]
MATERIALS = ["棉", "聚酯", "牛仔布", "羊毛", "亞麻", "絲綢"]
THICKNESS = ["薄", "中等", "厚"]
STYLES = ["休閒", "上班", "運動", "正式", "輕鬆"]

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

def predict_color_simple(image_path: str) -> str:
    """簡化的顏色預測 (基於像素分析)"""
    try:
        img = Image.open(image_path).convert("RGB").resize((50, 50))
        pixels = list(img.getdata())
        
        # 計算主要顏色通道
        r_avg = sum(p[0] for p in pixels) // len(pixels)
        g_avg = sum(p[1] for p in pixels) // len(pixels)
        b_avg = sum(p[2] for p in pixels) // len(pixels)
        
        # 簡單的顏色判斷
        if r_avg > 200 and g_avg > 200 and b_avg > 200:
            return "white"
        elif r_avg < 50 and g_avg < 50 and b_avg < 50:
            return "black"
        elif r_avg > g_avg and r_avg > b_avg:
            return "red"
        elif g_avg > r_avg and g_avg > b_avg:
            return "green"
        elif b_avg > r_avg and b_avg > g_avg:
            return "blue"
        else:
            return "gray"
    except:
        return "gray"

# ===== 主頁面 =====

st.title("👔 衣櫃穿搭推薦系統")

st.markdown("---")

# 側邊欄導航
with st.sidebar:
    st.title("🎯 菜單")
    page = st.radio("選擇功能", 
                   ["📊 首頁", "👕 添加衣服", "👀 查看衣櫃", 
                    "🎨 穿搭推薦", "🔍 搜尋衣服"])

# ===== 首頁 =====
if page == "📊 首頁":
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
            st.write("**按類別分布:**")
            cat_counts = {}
            for c in clothes:
                cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
            
            for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  • {cat}: {count} 件")
        
        with col2:
            st.write("**按風格分布:**")
            style_counts = {}
            for c in clothes:
                style_counts[c.style] = style_counts.get(c.style, 0) + 1
            
            for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  • {style}: {count} 件")
    else:
        st.info("📭 衣櫃還是空的，快去添加衣服吧！")

# ===== 添加衣服 =====
elif page == "👕 添加衣服":
    st.subheader("➕ 添加新衣服")
    
    tab1, tab2 = st.tabs(["📸 上傳圖片", "📝 手動輸入"])
    
    # Tab 1: 上傳圖片
    with tab1:
        st.info("📸 上傳衣服圖片，系統會自動分析顏色")
        
        uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 保存圖片
            image_path = os.path.join(IMAGE_DIR, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="上傳的圖片", use_column_width=True)
            
            with col2:
                st.write("🔍 正在分析...")
                
                # 預測顏色
                detected_color = predict_color_simple(image_path)
                st.success("✅ 分析完成!")
                st.write(f"**檢測到的顏色:** {COLOR_PALETTE.get(detected_color, detected_color)}")
            
            st.divider()
            
            st.write("📝 編輯詳情:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("衣服名稱", value=uploaded_file.name.split('.')[0])
                category = st.selectbox("類別", CATEGORIES)
                color = st.selectbox("顏色", list(COLOR_PALETTE.keys()),
                                    index=list(COLOR_PALETTE.keys()).index(detected_color)
                                    if detected_color in COLOR_PALETTE else 0)
            
            with col2:
                material = st.selectbox("材質", MATERIALS)
                thickness_val = st.selectbox("厚薄", THICKNESS)
                style = st.selectbox("風格", STYLES)
            
            if st.button("✅ 確認添加", use_container_width=True):
                add_clothing(
                    name=name,
                    category=category,
                    color=color,
                    material=material,
                    thickness=thickness_val,
                    style=style,
                    image_path=image_path
                )
                st.success(f"✅ 已添加: {name}")
                st.balloons()
    
    # Tab 2: 手動輸入
    with tab2:
        st.info("📝 手動填寫衣服信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("衣服名稱", key="manual_name")
            category = st.selectbox("類別", CATEGORIES, key="manual_cat")
            color = st.selectbox("顏色", list(COLOR_PALETTE.keys()), key="manual_color")
        
        with col2:
            material = st.selectbox("材質", MATERIALS, key="manual_mat")
            thickness_val = st.selectbox("厚薄", THICKNESS, key="manual_thick")
            style = st.selectbox("風格", STYLES, key="manual_style")
        
        if st.button("✅ 手動添加", use_container_width=True):
            if name:
                add_clothing(name, category, color, material, thickness_val, style)
                st.success(f"✅ 已添加: {name}")
                st.balloons()
            else:
                st.error("❌ 請輸入衣服名稱")

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
            "名稱": c.name,
            "類別": c.category,
            "顏色": COLOR_PALETTE.get(c.color, c.color),
            "厚薄": c.thickness,
            "風格": c.style,
            "材質": c.material,
            "ID": c.id
        } for c in filtered])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.write("🗑️ **刪除衣服**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_name = st.selectbox("選擇要刪除的衣服", 
                                        [f"{c.name} ({COLOR_PALETTE.get(c.color, c.color)})" for c in filtered],
                                        key="delete_select")
        
        with col2:
            if st.button("🗑️ 刪除", use_container_width=True):
                for c in filtered:
                    if f"{c.name} ({COLOR_PALETTE.get(c.color, c.color)})" == selected_name:
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
            style = st.selectbox("選擇風格", STYLES)
        
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
                    bottom_cats = [cat for cat in by_cat.keys() if cat in ["褲子", "裙子"]]
                    if bottom_cats:
                        outfit.append(by_cat[bottom_cats[0]][i % len(by_cat[bottom_cats[0]])])
                    
                    # 鞋子
                    if "鞋子" in by_cat:
                        outfit.append(by_cat["鞋子"][i % len(by_cat["鞋子"])])
                    
                    # 顯示穿搭
                    if outfit:
                        cols = st.columns(len(outfit))
                        for j, item in enumerate(outfit):
                            with cols[j]:
                                st.write(f"**{item.name}**")
                                st.write(f"顏色: {COLOR_PALETTE.get(item.color, item.color)}")
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
                "顏色": COLOR_PALETTE.get(c.color, c.color),
                "厚薄": c.thickness,
                "風格": c.style
            } for c in results])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"❌ 沒有找到包含 '{search_query}' 的衣服")
    else:
        st.info("輸入搜尋關鍵詞開始搜尋")

# ===== 底部 =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    👗 衣櫃穿搭推薦系統 v2.0 | Made with ❤️ by AI
</div>
""", unsafe_allow_html=True)
