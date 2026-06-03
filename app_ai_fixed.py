"""
Streamlit 完整版 - 真實 AI 衣服識別 + 天氣推薦
使用 CLIP 模型識別衣服 + 天氣 API 推薦穿搭

安裝: pip install streamlit requests pandas pillow matplotlib transformers torch torchvision
運行: streamlit run app_ai_fixed.py
"""

import streamlit as st
import os
import json
import uuid
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
    page_title="👔 AI 衣櫃穿搭推薦系統",
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
    .weather-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 配置 =====
BASE_DIR = "./closet_project"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CLOSET_FILE = os.path.join(BASE_DIR, "closet.json")

os.makedirs(IMAGE_DIR, exist_ok=True)

# 天氣 API - 已設定
WEATHER_API_KEY = "CWA-B72C4CEB-E263-4156-8387-1B216AE9DDE1"

# ===== 常數 =====

# CLIP 衣服標籤
CLOTHING_LABELS = [
    "a photo of a t-shirt", "a photo of a shirt", "a photo of a blouse",
    "a photo of a polo shirt", "a photo of a hoodie", "a photo of a sweater",
    "a photo of a jacket", "a photo of a coat", "a photo of a blazer",
    "a photo of a dress", "a photo of a skirt", "a photo of pants",
    "a photo of jeans", "a photo of shorts", "a photo of sneakers",
    "a photo of shoes", "a photo of sandals", "a photo of boots",
    "a photo of a bag", "a photo of a hat", "a photo of a vest",
    "a photo of cargo pants", "a photo of leggings"
]

# 衣服標籤到類別的映射
CLOTHING_CATEGORY_MAP = {
    "t-shirt": "上衣", "shirt": "上衣", "blouse": "上衣", "polo": "上衣",
    "hoodie": "上衣", "sweater": "上衣", "vest": "上衣",
    "jacket": "外套", "coat": "外套", "blazer": "外套",
    "pants": "褲子", "jeans": "牛仔褲", "shorts": "短褲", 
    "skirt": "裙子", "leggings": "褲襪",
    "dress": "連衣裙",
    "sneakers": "運動鞋", "shoes": "鞋子", "sandals": "涼鞋", "boots": "靴子",
    "bag": "包包", "hat": "帽子"
}

COLOR_PALETTE = {
    "black": "黑色", "white": "白色", "gray": "灰色",
    "red": "紅色", "blue": "藍色", "green": "綠色",
    "yellow": "黃色", "brown": "棕色", "pink": "粉紅色",
    "purple": "紫色", "orange": "橙色", "beige": "米色",
}

MATERIALS = ["棉", "聚酯", "牛仔布", "羊毛", "亞麻", "絲綢", "棉麻混紡"]
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

# ===== CLIP 模型 (緩存載入) =====

@st.cache_resource
def load_clip_model():
    """載入 CLIP 模型 (只載入一次)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    progress_placeholder = st.empty()
    progress_placeholder.info(f"🤖 正在載入 CLIP 模型... (使用 {device.upper()}) 這需要約 30 秒，只需一次")
    
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        progress_placeholder.success(f"✅ CLIP 模型已載入!")
        return processor, model, device
    except Exception as e:
        progress_placeholder.error(f"❌ 模型載入失敗: {e}")
        return None, None, None

# ===== 衣服識別函數 =====

def predict_clothing_label(image_path: str) -> Tuple[str, float, str]:
    """
    預測衣服標籤
    返回: (標籤, 信心度, 類別)
    """
    try:
        processor, model, device = load_clip_model()
        
        if model is None:
            return "unknown", 0.0, "上衣"
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=CLOTHING_LABELS, images=image, 
                         return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        best_idx = int(probs.argmax())
        best_label = CLOTHING_LABELS[best_idx]
        best_score = float(probs[best_idx])
        
        # 轉換為類別
        category = map_label_to_category(best_label)
        
        return best_label, best_score, category
    except Exception as e:
        st.error(f"❌ 預測失敗: {e}")
        return "unknown", 0.0, "上衣"

def map_label_to_category(label: str) -> str:
    """將標籤對應到類別"""
    text = label.lower()
    
    for keyword, category in CLOTHING_CATEGORY_MAP.items():
        if keyword in text:
            return category
    
    return "上衣"  # 預設

def predict_color(image_path: str) -> Tuple[str, str]:
    """預測主色調"""
    try:
        img = Image.open(image_path).convert("RGB").resize((100, 100))
        pixels = list(img.getdata())
        
        # 過濾背景
        filtered = [p for p in pixels if not (p[0] > 245 and p[1] > 245 and p[2] > 245)]
        if not filtered:
            filtered = pixels
        
        # 計算平均顏色
        r_avg = sum(p[0] for p in filtered) // len(filtered)
        g_avg = sum(p[1] for p in filtered) // len(filtered)
        b_avg = sum(p[2] for p in filtered) // len(filtered)
        
        # 顏色判斷
        if r_avg > 200 and g_avg > 200 and b_avg > 200:
            detected_color = "white"
        elif r_avg < 50 and g_avg < 50 and b_avg < 50:
            detected_color = "black"
        elif r_avg > g_avg and r_avg > b_avg:
            detected_color = "red"
        elif g_avg > r_avg and g_avg > b_avg:
            detected_color = "green"
        elif b_avg > r_avg and b_avg > g_avg:
            detected_color = "blue"
        elif (r_avg + g_avg) // 2 > b_avg:
            detected_color = "yellow"
        else:
            detected_color = "gray"
        
        return detected_color, COLOR_PALETTE.get(detected_color, detected_color)
    except:
        return "gray", "灰色"

def infer_material_thickness(category: str, label: str) -> Tuple[str, str]:
    """根據衣服類別推斷材質和厚薄"""
    label_lower = label.lower()
    material, thickness = "棉", "中等"
    
    if any(x in label_lower for x in ["coat", "blazer"]):
        thickness, material = "厚", "聚酯"
    elif any(x in label_lower for x in ["jacket", "vest"]):
        thickness, material = "中等", "聚酯"
    elif any(x in label_lower for x in ["t-shirt", "shirt", "blouse", "polo"]):
        thickness, material = "薄", "棉"
    elif any(x in label_lower for x in ["hoodie", "sweater"]):
        thickness, material = "中等", "棉"
    elif "jeans" in label_lower:
        thickness, material = "中等", "牛仔布"
    elif any(x in label_lower for x in ["shorts", "skirt"]):
        thickness, material = "薄", "棉"
    elif "dress" in label_lower:
        thickness, material = "薄", "棉"
    
    return material, thickness

# ===== 天氣 API =====

def get_weather(city: str = "臺中市") -> Optional[Dict]:
    """
    獲取天氣信息
    返回: {temp, humidity, weather_type, description}
    """
    try:
        params = {
            "Authorization": WEATHER_API_KEY,
            "locationName": city,
            "elementName": "Temperature,RelativeHumidity,Weather"
        }
        
        response = requests.get(
            "https://opendata.cwa.gov.tw/api/v1/rest/ObsCurrentStatus",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("records", {}).get("location"):
            location = data["records"]["location"][0]
            weather_elements = {
                elem["elementName"]: elem["elementValue"]
                for elem in location.get("weatherElement", [])
            }
            
            temp = float(weather_elements.get("Temperature", 20))
            humidity = float(weather_elements.get("RelativeHumidity", 50))
            weather_type = weather_elements.get("Weather", "晴天")
            
            return {
                "temp": temp,
                "humidity": humidity,
                "weather_type": weather_type,
                "city": city,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        st.warning(f"⚠️ 獲取天氣失敗: {e}")
    
    return None

def recommend_thickness_by_weather(weather: Dict) -> str:
    """根據天氣推薦厚薄"""
    if not weather:
        return "中等"
    
    temp = weather.get("temp", 20)
    weather_type = weather.get("weather_type", "").lower()
    
    # 下雨/風大需要厚一點
    if any(x in weather_type for x in ["rain", "雨", "wind", "風"]):
        if temp < 10:
            return "厚"
        elif temp < 18:
            return "中等"
    
    # 根據溫度
    if temp < 5:
        return "厚"
    elif temp < 15:
        return "中等"
    else:
        return "薄"

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

# ===== 主頁面 =====

st.title("👔 AI 衣櫃穿搭推薦系統")
st.markdown("🤖 *使用 CLIP 模型識別衣服 + 天氣 API 推薦穿搭*")
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.title("🎯 菜單")
    page = st.radio("選擇功能", 
                   ["📊 首頁", "👕 添加衣服", "👀 查看衣櫃", 
                    "🎨 天氣穿搭推薦", "🔍 搜尋衣服", "ℹ️ 關於"])

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
    
    tab1, tab2 = st.tabs(["📸 AI 識別", "📝 手動輸入"])
    
    # Tab 1: AI 識別
    with tab1:
        st.info("📸 上傳衣服圖片，AI 會自動識別類別和顏色")
        
        uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"], key="upload_ai")
        
        if uploaded_file is not None:
            # 保存圖片
            image_path = os.path.join(IMAGE_DIR, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="上傳的圖片", use_column_width=True)
            
            with col2:
                st.write("🔍 正在 AI 分析...")
                
                with st.spinner("分析中... (第一次需要載入模型，約 30 秒)"):
                    # AI 預測
                    label, confidence, category = predict_clothing_label(image_path)
                    color, color_cn = predict_color(image_path)
                    material, thickness_val = infer_material_thickness(category, label)
                    
                    st.success("✅ 分析完成!")
                    
                    st.write(f"**✨ AI 識別結果:**")
                    st.write(f"  • 衣服類型: {label}")
                    st.write(f"  • 信心度: {confidence:.1%} 🎯")
                    st.write(f"  • 類別: **{category}**")
                    st.write(f"  • 顏色: **{color_cn}**")
                    st.write(f"  • 推估厚薄: **{thickness_val}**")
                    st.write(f"  • 推估材質: **{material}**")
            
            st.divider()
            
            st.write("📝 可以編輯詳情:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("衣服名稱", value=uploaded_file.name.split('.')[0])
                categories_list = list(set(CLOTHING_CATEGORY_MAP.values()))
                category_select = st.selectbox("類別", categories_list,
                                              index=categories_list.index(category) if category in categories_list else 0)
                color_keys = list(COLOR_PALETTE.keys())
                color_select = st.selectbox("顏色", color_keys,
                                           index=color_keys.index(color) if color in color_keys else 0)
            
            with col2:
                material_select = st.selectbox("材質", MATERIALS, 
                                              index=MATERIALS.index(material) if material in MATERIALS else 0)
                thickness_select = st.selectbox("厚薄", THICKNESS,
                                               index=THICKNESS.index(thickness_val) if thickness_val in THICKNESS else 0)
                style_select = st.selectbox("風格", STYLES)
            
            if st.button("✅ 確認添加", use_container_width=True, key="add_ai"):
                add_clothing(
                    name=name,
                    category=category_select,
                    color=color_select,
                    material=material_select,
                    thickness=thickness_select,
                    style=style_select,
                    image_path=image_path
                )
                st.success(f"✅ 已添加: {name}")
                st.balloons()
                st.rerun()
    
    # Tab 2: 手動輸入
    with tab2:
        st.info("📝 手動填寫衣服信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("衣服名稱", key="manual_name")
            category = st.selectbox("類別", list(set(CLOTHING_CATEGORY_MAP.values())), key="manual_cat")
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
                st.rerun()
            else:
                st.error("❌ 請輸入衣服名稱")

# ===== 查看衣櫃 =====
elif page == "👀 查看衣櫃":
    st.subheader("📦 我的衣櫃")
    
    clothes = load_closet()
    
    if not clothes:
        st.info("衣櫃還是空的，快去添加衣服吧！")
    else:
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
        
        filtered = clothes
        if filter_category != "全部":
            filtered = [c for c in filtered if c.category == filter_category]
        if filter_color != "全部":
            filtered = [c for c in filtered if c.color == filter_color]
        if filter_style != "全部":
            filtered = [c for c in filtered if c.style == filter_style]
        
        st.write(f"**找到 {len(filtered)} 件衣服**")
        
        df = pd.DataFrame([{
            "ID": c.id,
            "名稱": c.name,
            "類別": c.category,
            "顏色": COLOR_PALETTE.get(c.color, c.color),
            "厚薄": c.thickness,
            "風格": c.style,
            "材質": c.material
        } for c in filtered])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.write("🗑️ **刪除衣服**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_idx = st.selectbox("選擇要刪除的衣服", 
                                        range(len(filtered)),
                                        format_func=lambda i: f"{filtered[i].name} ({COLOR_PALETTE.get(filtered[i].color, filtered[i].color)})",
                                        key="delete_select")
        
        with col2:
            if st.button("🗑️ 刪除", use_container_width=True):
                delete_clothing(filtered[selected_idx].id)
                st.success("✅ 已刪除")
                st.rerun()

# ===== 天氣穿搭推薦 =====
elif page == "🎨 天氣穿搭推薦":
    st.subheader("🌤️ 根據天氣推薦穿搭")
    
    clothes = load_closet()
    
    if not clothes:
        st.warning("衣櫃為空，無法推薦穿搭")
    else:
        # 選擇城市
        city = st.text_input("輸入城市名稱", value="臺中市", help="例: 臺北市、高雄市、臺中市、台南市、屏東縣")
        
        if st.button("🌤️ 獲取天氣並推薦穿搭", use_container_width=True):
            with st.spinner("正在獲取天氣..."):
                weather = get_weather(city)
            
            if weather:
                # 顯示天氣信息
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🌡️ 溫度", f"{weather['temp']:.1f}°C")
                
                with col2:
                    st.metric("💧 濕度", f"{weather['humidity']:.0f}%")
                
                with col3:
                    st.metric("⛅ 天氣", weather['weather_type'])
                
                with col4:
                    thickness_rec = recommend_thickness_by_weather(weather)
                    st.metric("👕 推薦厚薄", thickness_rec)
                
                st.markdown("---")
                
                # 根據天氣推薦穿搭
                st.write("### 👗 推薦穿搭方案")
                
                # 根據天氣推薦厚薄
                recommended_thickness = recommend_thickness_by_weather(weather)
                
                # 篩選衣服
                tops = [c for c in clothes if c.category == "上衣"]
                bottoms = [c for c in clothes if c.category in ["褲子", "牛仔褲", "短褲", "裙子", "連衣裙"]]
                shoes = [c for c in clothes if c.category == "鞋子"]
                
                if not tops or not bottoms:
                    st.warning("⚠️ 衣櫃缺少必要的衣服 (上衣或下裝)")
                else:
                    # 生成多個穿搭建議
                    num_outfits = st.slider("推薦套數", 1, 5, 3)
                    
                    for i in range(num_outfits):
                        st.write(f"#### 穿搭方案 {i+1}")
                        
                        # 優先選擇符合厚薄的衣服
                        filtered_tops = [c for c in tops if c.thickness == recommended_thickness]
                        if not filtered_tops:
                            filtered_tops = tops
                        
                        outfit_top = filtered_tops[i % len(filtered_tops)]
                        outfit_bottom = bottoms[i % len(bottoms)]
                        outfit_shoes = shoes[i % len(shoes)] if shoes else None
                        
                        cols = st.columns([1, 1, 1] if outfit_shoes else [1, 1])
                        
                        with cols[0]:
                            st.write(f"**{outfit_top.name}**")
                            st.write(f"🎨 {COLOR_PALETTE.get(outfit_top.color, outfit_top.color)}")
                            st.write(f"📏 {outfit_top.thickness}")
                        
                        with cols[1]:
                            st.write(f"**{outfit_bottom.name}**")
                            st.write(f"🎨 {COLOR_PALETTE.get(outfit_bottom.color, outfit_bottom.color)}")
                            st.write(f"📏 {outfit_bottom.thickness}")
                        
                        if outfit_shoes:
                            with cols[2]:
                                st.write(f"**{outfit_shoes.name}**")
                                st.write(f"🎨 {COLOR_PALETTE.get(outfit_shoes.color, outfit_shoes.color)}")
                                st.write(f"👟 {outfit_shoes.thickness}")
                        
                        st.divider()
            else:
                st.error(f"❌ 無法獲取 {city} 的天氣信息")
                st.info("💡 提示: 請嘗試使用正式的城市名稱，如「臺北市」、「新北市」、「臺中市」等")

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

# ===== 關於 =====
elif page == "ℹ️ 關於":
    st.subheader("ℹ️ 關於此系統")
    
    st.markdown("""
    ### 👔 AI 衣櫃穿搭推薦系統 v3.0
    
    一個智慧的個人衣櫃管理和穿搭推薦系統，使用最新的 AI 技術。
    
    #### 🌟 核心功能
    
    - **🤖 AI 衣服識別**: 使用 OpenAI CLIP 模型自動識別衣服類別
    - **🌤️ 天氣智慧推薦**: 根據實時天氣推薦穿搭方案
    - **📊 衣櫃管理**: 完整的添加、查看、搜尋、刪除功能
    - **🎨 色彩搭配**: 自動檢測衣服顏色
    
    #### 🔧 技術棧
    
    - **Streamlit**: 互動式網頁界面
    - **CLIP 模型**: OpenAI 視覺語言模型
    - **天氣 API**: 中央氣象署公開數據
    - **PyTorch**: 深度學習推理
    
    #### 📱 使用流程
    
    1. **添加衣服** 👕
       - 上傳衣服圖片
       - AI 自動識別類別、顏色、厚薄
       - 確認添加到衣櫃
    
    2. **天氣推薦** 🌤️
       - 輸入城市名稱
       - 獲取實時天氣
       - 根據溫度/天氣推薦穿搭
    
    3. **查看衣櫃** 👀
       - 表格展示所有衣服
       - 按類別、顏色、風格篩選
       - 快速刪除舊衣服
    
    #### 💾 數據存儲
    
    所有數據存儲在本地:
    - `./closet_project/closet.json` - 衣服信息
    - `./closet_project/images/` - 衣服圖片
    
    #### 🎯 設計理念
    
    - 簡單易用: 無需技術知識
    - 智慧推薦: 基於天氣和衣服屬性
    - 隱私保護: 數據存儲在本地
    - 開源免費: 完全開源
    
    #### 📝 版本歷史
    
    - **v3.0** (當前): AI 識別 + 天氣推薦完整版
    - **v2.0**: 模組化架構版本
    - **v1.0**: 初始 Colab 版本
    
    #### 🤝 貢獻
    
    歡迎提交 Issue 和 Pull Request!
    
    GitHub: https://github.com/yuze06/python
    
    #### 📧 聯繫方式
    
    Email: ya0970731775@gmail.com
    
    ---
    
    **Made with ❤️ by AI Fashion Assistant**
    """)

# ===== 底部 =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    👗 AI 衣櫃穿搭推薦系統 v3.0 | 使用 CLIP + 天氣 API | Made with ❤️ by AI
</div>
""", unsafe_allow_html=True)
