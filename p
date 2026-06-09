python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
pip install flask flask-cors pillow requests anthropic python-dotenv

ANTHROPIC_API_KEY=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
OPENWEATHER_API_KEY=f00d663a3c78924f34b39def90c389e8
DEFAULT_CITY=Taipei
FLASK_ENV=development

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import requests
from anthropic import Anthropic

# ========== 環境設定 ==========
app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
DEFAULT_CITY = os.getenv('DEFAULT_CITY', 'Taipei')

client = Anthropic()

# ========== 資料庫設定 ==========
DB_PATH = 'wardrobe.db'

def init_db():
    """初始化資料庫"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS clothing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clothing_type TEXT NOT NULL,
            color TEXT NOT NULL,
            style TEXT NOT NULL,
            suitable_weather TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ========== 天氣 API ==========
@app.route('/api/weather', methods=['GET'])
def get_weather():
    """取得天氣資訊"""
    city = request.args.get('city', DEFAULT_CITY)
    
    try:
        # 呼叫 OpenWeather API
        url = f'https://api.openweathermap.org/data/2.5/weather'
        params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'lang': 'zh_tw'
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # 判斷氣溫級別
        temp = data['main']['temp']
        if temp < 10:
            temp_level = '很冷'
        elif temp < 15:
            temp_level = '寒冷'
        elif temp < 20:
            temp_level = '涼爽'
        elif temp < 25:
            temp_level = '舒適'
        elif temp < 30:
            temp_level = '溫暖'
        else:
            temp_level = '炎熱'
        
        # 天氣圖標映射
        icon_map = {
            '01d': '☀️', '01n': '🌙',
            '02d': '⛅', '02n': '☁️',
            '03d': '☁️', '03n': '☁️',
            '04d': '☁️', '04n': '☁️',
            '09d': '🌧️', '09n': '🌧️',
            '10d': '🌦️', '10n': '🌧️',
            '11d': '⛈️', '11n': '⛈️',
            '13d': '❄️', '13n': '❄️',
            '50d': '🌫️', '50n': '🌫️'
        }
        
        icon_code = data['weather'][0]['icon']
        emoji_icon = icon_map.get(icon_code, '🌤️')
        
        return jsonify({
            'city': data['name'],
            'temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'humidity': data['main']['humidity'],
            'wind_speed': round(data['wind']['speed'], 1),
            'description': data['weather'][0]['main'],
            'icon': emoji_icon,
            'temp_level': temp_level,
            'is_mock': False
        })
    
    except Exception as e:
        print(f"天氣 API 錯誤: {e}")
        # 返回測試數據
        return jsonify({
            'city': city,
            'temp': 22,
            'feels_like': 20,
            'humidity': 65,
            'wind_speed': 3.5,
            'description': '多雲',
            'icon': '⛅',
            'temp_level': '舒適',
            'is_mock': True
        })

# ========== 衣服上傳與 AI 識別 ==========
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

def encode_image_to_base64(image_data):
    """將圖片編碼為 Base64"""
    img = Image.open(image_data)
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    return base64.standard_b64encode(buffer.read()).decode('utf-8')

@app.route('/api/upload', methods=['POST'])
def upload_clothing():
    """上傳衣服圖片並用 AI 識別"""
    if 'file' not in request.files:
        return jsonify({'error': '未提供圖片'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未選擇圖片'}), 400
    
    try:
        # 編碼圖片
        image_base64 = encode_image_to_base64(file)
        
        # 呼叫 Claude AI 識別衣服
        message = client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=500,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': image_base64
                            }
                        },
                        {
                            'type': 'text',
                            'text': '''請分析這件衣服，並用繁體中文回答以下問題，格式如下（每行一個）：
類型: [衣服類型，如：T恤、牛仔褲、外套等]
顏色: [主要顏色]
風格: [風格，如：休閒、正式、運動、波希米亞等]
適合天氣: [如：炎熱、溫暖、涼爽、寒冷等]

只需要回答這四行，不要其他內容。'''
                        }
                    ]
                }
            ]
        )
        
        # 解析 AI 回應
        response_text = message.content[0].text
        lines = response_text.strip().split('\n')
        
        result = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        # 保存圖片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{timestamp}_{file.filename}'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 保存到資料庫
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO clothing (clothing_type, color, style, suitable_weather, image_path, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.get('類型', '未知'),
            result.get('顏色', '未知'),
            result.get('風格', '未知'),
            result.get('適合天氣', '多季節'),
            filepath,
            f'/uploads/{filename}'
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'clothing_type': result.get('類型', '未知'),
            'color': result.get('顏色', '未知'),
            'style': result.get('風格', '未知'),
            'suitable_weather': result.get('適合天氣', '多季節')
        })
    
    except Exception as e:
        print(f"上傳錯誤: {e}")
        return jsonify({'error': f'識別失敗: {str(e)}'}), 500

# ========== 衣櫥管理 ==========
@app.route('/api/wardrobe', methods=['GET'])
def get_wardrobe():
    """取得衣櫥列表"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM clothing ORDER BY created_at DESC')
        items = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(items)
    except Exception as e:
        print(f"取得衣櫥錯誤: {e}")
        return jsonify([])

@app.route('/api/wardrobe/<int:item_id>', methods=['DELETE'])
def delete_wardrobe_item(item_id):
    """刪除衣櫥項目"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 取得圖片路徑
        c.execute('SELECT image_path FROM clothing WHERE id = ?', (item_id,))
        row = c.fetchone()
        
        if row and row[0]:
            image_path = row[0]
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # 刪除資料庫記錄
        c.execute('DELETE FROM clothing WHERE id = ?', (item_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"刪除錯誤: {e}")
        return jsonify({'error': '刪除失敗'}), 500

# ========== 穿搭建議 ==========
@app.route('/api/suggest', methods=['POST'])
def get_suggestion():
    """取得 AI 穿搭建議"""
    data = request.json
    city = data.get('city', DEFAULT_CITY)
    
    try:
        # 取得天氣
        weather_url = f'https://api.openweathermap.org/data/2.5/weather'
        weather_params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'lang': 'zh_tw'
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=5)
        weather_data = weather_response.json()
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['main']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        
        # 取得衣櫥列表
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM clothing ORDER BY created_at DESC LIMIT 20')
        wardrobe_items = [dict(row) for row in c.fetchall()]
        conn.close()
        
        if not wardrobe_items:
            return jsonify({'suggestion': '衣櫥空的，無法生成建議。請先上傳一些衣服！'})
        
        # 組織衣櫥信息
        wardrobe_text = '你的衣櫥有以下衣服：\n'
        for i, item in enumerate(wardrobe_items, 1):
            wardrobe_text += f"{i}. {item['clothing_type']}（{item['color']}，{item['style']}風格，適合{item['suitable_weather']}天氣）\n"
        
        # 呼叫 Claude AI 生成建議
        message = client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=1000,
            messages=[
                {
                    'role': 'user',
                    'content': f'''根據以下天氣和用戶的衣櫥，請用繁體中文提供今日穿搭建議。

【今日天氣】
城市: {city}
氣溫: {round(temp)}°C
天氣: {description}
濕度: {humidity}%
風速: {wind_speed} m/s

【用戶衣櫥】
{wardrobe_text}

請提供具體的穿搭建議，包括：
1. 推薦的衣服組合
2. 搭配理由（基於溫度、天氣等）
3. 額外建議（如配件、鞋子等）

建議要實用、友善、鼓勵性的語氣，字數控制在 200-300 字。'''
                }
            ]
        )
        
        suggestion = message.content[0].text
        return jsonify({'suggestion': suggestion})
    
    except Exception as e:
        print(f"穿搭建議錯誤: {e}")
        return jsonify({'error': f'生成建議失敗: {str(e)}'}), 500

# ========== 靜態文件服務 ==========
@app.route('/uploads/<filename>')
def serve_upload(filename):
    """提供上傳圖片的靜態文件"""
    return app.send_static_file(f'uploads/{filename}')

# ========== 首頁路由 ==========
@app.route('/')
def index():
    """提供 HTML 前端"""
    # 這裡返回您提供的 HTML 代碼
    html = '''<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>今日穿搭助理</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=DM+Serif+Display&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --sky: #C8D8E8;
      --sky-dark: #8FAEC8;
      --ink: #1A1F2E;
      --ink-soft: #4A5568;
      --cream: #F7F4EF;
      --warm: #E8D5B0;
      --accent: #D4764A;
      --accent-light: #F0C4A8;
      --green: #5A8A6A;
      --card-bg: #FFFFFF;
      --border: #E2DDD6;
      --shadow: 0 2px 20px rgba(26,31,46,0.08);
      --radius: 16px;
    }

    body {
      font-family: 'Noto Sans TC', sans-serif;
      background: var(--cream);
      color: var(--ink);
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(135deg, var(--sky) 0%, var(--sky-dark) 100%);
      padding: 28px 24px 24px;
      position: relative;
      overflow: hidden;
    }
    header::after {
      content: '';
      position: absolute;
      bottom: -30px; left: 0; right: 0;
      height: 60px;
      background: var(--cream);
      border-radius: 50% 50% 0 0 / 60px 60px 0 0;
    }
    .header-inner {
      max-width: 900px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }
    .logo {
      font-family: 'DM Serif Display', serif;
      font-size: 26px;
      color: var(--ink);
      letter-spacing: -0.5px;
    }
    .logo span { color: var(--accent); }
    .city-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(255,255,255,0.8);
      border-radius: 40px;
      padding: 6px 14px;
      backdrop-filter: blur(8px);
    }
    .city-bar input {
      border: none; background: transparent;
      font-family: inherit; font-size: 14px;
      color: var(--ink); width: 120px; outline: none;
    }
    .city-bar button {
      background: var(--accent); color: white;
      border: none; border-radius: 20px;
      padding: 4px 12px; font-size: 13px;
      cursor: pointer; transition: opacity .2s;
    }
    .city-bar button:hover { opacity: .85; }

    /* ── Main Layout ── */
    main {
      max-width: 900px;
      margin: 0 auto;
      padding: 20px 16px 60px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 640px) { main { grid-template-columns: 1fr; } }

    /* ── Cards ── */
    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card-header {
      padding: 18px 20px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-icon {
      width: 32px; height: 32px;
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--ink);
    }
    .card-body { padding: 16px 20px 20px; }

    /* ── Weather Card ── */
    #weather-card { grid-column: 1 / -1; }
    .weather-main {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 12px 0;
    }
    .weather-icon img { width: 64px; }
    .weather-temp {
      font-family: 'DM Serif Display', serif;
      font-size: 52px;
      line-height: 1;
      color: var(--ink);
    }
    .weather-temp sup { font-size: 20px; font-family: inherit; }
    .weather-details {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }
    .weather-pill {
      background: var(--cream);
      border-radius: 20px;
      padding: 5px 12px;
      font-size: 13px;
      color: var(--ink-soft);
    }
    .weather-pill strong { color: var(--ink); }
    .weather-mock-badge {
      background: var(--accent-light);
      color: var(--accent);
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
    }
    .weather-loading {
      padding: 20px 0;
      color: var(--ink-soft);
      font-size: 14px;
    }

    /* ── Upload Card ── */
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 32px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(212,118,74,0.04);
    }
    .upload-zone input { display: none; }
    .upload-zone .upload-emoji { font-size: 32px; margin-bottom: 8px; }
    .upload-zone p { font-size: 13px; color: var(--ink-soft); }
    .upload-zone p strong { color: var(--accent); }

    .upload-preview {
      display: none;
      margin-top: 14px;
      position: relative;
    }
    .upload-preview img {
      width: 100%;
      max-height: 200px;
      object-fit: contain;
      border-radius: 10px;
    }
    .upload-preview-cancel {
      position: absolute; top: 6px; right: 6px;
      background: var(--ink); color: white;
      border: none; border-radius: 50%;
      width: 24px; height: 24px;
      font-size: 14px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
    }

    .btn {
      display: block; width: 100%;
      background: var(--accent); color: white;
      border: none; border-radius: 12px;
      padding: 13px;
      font-family: inherit; font-size: 15px; font-weight: 700;
      cursor: pointer;
      transition: transform .15s, opacity .2s;
      margin-top: 14px;
    }
    .btn:hover { opacity: .9; transform: translateY(-1px); }
    .btn:disabled { opacity: .5; cursor: not-allowed; transform: none; }
    .btn-secondary {
      background: var(--cream);
      color: var(--ink);
      border: 1px solid var(--border);
    }

    /* Status messages */
    .status-msg {
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 13px;
      display: none;
    }
    .status-msg.success { background: #EAF5EC; color: var(--green); display: block; }
    .status-msg.error   { background: #FDEEEA; color: #C0452A; display: block; }
    .status-msg.loading { background: var(--cream); color: var(--ink-soft); display: block; }

    /* ── Wardrobe ── */
    #wardrobe-card { grid-column: 1 / -1; }
    .wardrobe-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 12px;
      margin-top: 4px;
    }
    .wardrobe-item {
      background: var(--cream);
      border-radius: 12px;
      overflow: hidden;
      position: relative;
      transition: transform .2s;
    }
    .wardrobe-item:hover { transform: translateY(-3px); }
    .wardrobe-item img {
      width: 100%; height: 120px;
      object-fit: cover; display: block;
    }
    .wardrobe-item-info {
      padding: 8px 10px;
    }
    .wardrobe-item-type {
      font-size: 12px; font-weight: 700; color: var(--ink);
    }
    .wardrobe-item-color {
      font-size: 11px; color: var(--ink-soft);
    }
    .wardrobe-item-tag {
      display: inline-block;
      background: var(--accent-light);
      color: var(--accent);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      font-weight: 700;
      margin-top: 3px;
    }
    .wardrobe-delete {
      position: absolute; top: 6px; right: 6px;
      background: rgba(26,31,46,0.6);
      color: white; border: none; border-radius: 50%;
      width: 22px; height: 22px;
      font-size: 12px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      opacity: 0; transition: opacity .2s;
    }
    .wardrobe-item:hover .wardrobe-delete { opacity: 1; }
    .wardrobe-empty {
      text-align: center;
      padding: 30px 0;
      color: var(--ink-soft);
      font-size: 14px;
    }

    /* ── Suggestion Card ── */
    #suggest-card { grid-column: 1 / -1; }
    .suggestion-text {
      font-size: 14px;
      line-height: 1.8;
      color: var(--ink-soft);
      white-space: pre-wrap;
      padding: 4px 0;
    }
    .suggestion-placeholder {
      text-align: center;
      padding: 20px 0;
      color: var(--ink-soft);
      font-size: 14px;
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.4);
      border-top-color: white;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="logo">今日<span>穿搭</span>助理</div>
    <div class="city-bar">
      <span>📍</span>
      <input type="text" id="city-input" value="Taipei" placeholder="輸入城市">
      <button onclick="loadWeather()">更新</button>
    </div>
  </div>
</header>

<main>

  <!-- 天氣卡片 -->
  <div class="card" id="weather-card">
    <div class="card-header">
      <div class="card-icon" style="background:#EBF4FB">☀️</div>
      <span class="card-title">今日天氣</span>
      <span id="mock-badge" class="weather-mock-badge" style="display:none;margin-left:auto">測試模式</span>
    </div>
    <div class="card-body">
      <div id="weather-loading" class="weather-loading">載入天氣中…</div>
      <div id="weather-content" style="display:none">
        <div class="weather-main">
          <div class="weather-icon"><span id="w-icon">☀️</span></div>
          <div class="weather-temp"><span id="w-temp">--</span><sup>°C</sup></div>
          <div>
            <div style="font-size:16px;font-weight:700" id="w-desc">--</div>
            <div style="font-size:13px;color:var(--ink-soft);margin-top:4px" id="w-city">--</div>
          </div>
        </div>
        <div class="weather-details">
          <div class="weather-pill">體感 <strong id="w-feels">--</strong>°C</div>
          <div class="weather-pill">濕度 <strong id="w-humidity">--</strong>%</div>
          <div class="weather-pill">風速 <strong id="w-wind">--</strong> m/s</div>
          <div class="weather-pill">氣候 <strong id="w-level">--</strong></div>
        </div>
      </div>
    </div>
  </div>

  <!-- 上傳衣服 -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#FDF0E8">👔</div>
      <span class="card-title">新增衣服</span>
    </div>
    <div class="card-body">
      <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
        <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
        <div class="upload-emoji">📸</div>
        <p>點擊或拖拽圖片到這裡</p>
        <p style="margin-top:4px"><strong>支援 JPG、PNG、WEBP</strong></p>
      </div>
      <div class="upload-preview" id="upload-preview">
        <img id="preview-img" src="" alt="預覽">
        <button class="upload-preview-cancel" onclick="cancelUpload()">✕</button>
      </div>
      <div id="upload-status" class="status-msg"></div>
      <button class="btn" id="upload-btn" onclick="uploadClothing()" disabled>
        用 AI 識別並加入衣櫥
      </button>
    </div>
  </div>

  <!-- 穿搭建議 -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#EAF5EC">✨</div>
      <span class="card-title">今日穿搭建議</span>
    </div>
    <div class="card-body">
      <div id="suggest-placeholder" class="suggestion-placeholder">
        <div style="font-size:36px;margin-bottom:8px">🪄</div>
        點擊下方按鈕，根據今天天氣為你搭配穿著
      </div>
      <div id="suggest-result" class="suggestion-text" style="display:none"></div>
      <div id="suggest-status" class="status-msg"></div>
      <button class="btn" id="suggest-btn" onclick="getSuggestion()">
        為我搭配今日穿搭
      </button>
    </div>
  </div>

  <!-- 我的衣櫥 -->
  <div class="card" id="wardrobe-card">
    <div class="card-header">
      <div class="card-icon" style="background:#F5F0F8">👗</div>
      <span class="card-title">我的衣櫥</span>
      <span id="wardrobe-count" style="margin-left:auto;font-size:12px;color:var(--ink-soft)">0 件</span>
    </div>
    <div class="card-body">
      <div id="wardrobe-grid" class="wardrobe-grid">
        <div class="wardrobe-empty">衣櫥是空的，快來上傳衣服吧！</div>
      </div>
    </div>
  </div>

</main>

<script>
let currentWeather = null;
let selectedFile = null;

// ── 天氣 ──────────────────────────────
async function loadWeather() {
  const city = document.getElementById('city-input').value.trim() || 'Taipei';
  document.getElementById('weather-loading').style.display = 'block';
  document.getElementById('weather-content').style.display = 'none';

  try {
    const res = await fetch(`/api/weather?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    currentWeather = data;

    document.getElementById('w-icon').textContent = data.icon;
    document.getElementById('w-temp').textContent = data.temp;
    document.getElementById('w-desc').textContent = data.description;
    document.getElementById('w-city').textContent = data.city;
    document.getElementById('w-feels').textContent = data.feels_like;
    document.getElementById('w-humidity').textContent = data.humidity;
    document.getElementById('w-wind').textContent = data.wind_speed;
    document.getElementById('w-level').textContent = data.temp_level;

    document.getElementById('mock-badge').style.display = data.is_mock ? 'inline' : 'none';
    document.getElementById('weather-loading').style.display = 'none';
    document.getElementById('weather-content').style.display = 'block';
  } catch (e) {
    document.getElementById('weather-loading').textContent = '天氣載入失敗，請確認城市名稱';
  }
}

// ── 上傳 ──────────────────────────────
const uploadZone = document.getElementById('upload-zone');
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) previewFile(file);
});

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) previewFile(file);
}

function previewFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('upload-preview').style.display = 'block';
    document.getElementById('upload-btn').disabled = false;
    setStatus('upload-status', '', '');
  };
  reader.readAsDataURL(file);
}

function cancelUpload() {
  selectedFile = null;
  document.getElementById('upload-preview').style.display = 'none';
  document.getElementById('upload-btn').disabled = true;
  document.getElementById('file-input').value = '';
  setStatus('upload-status', '', '');
}

async function uploadClothing() {
  if (!selectedFile) return;

  const btn = document.getElementById('upload-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>AI 識別中…';
  setStatus('upload-status', 'loading', '正在用 AI 識別衣服種類，請稍候…');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      setStatus('upload-status', 'error', '❌ ' + data.error);
    } else {
      setStatus('upload-status', 'success',
        `✅ 識別成功！${data.clothing_type}（${data.color}，${data.style}風格）`);
      cancelUpload();
      loadWardrobe();
    }
  } catch (e) {
    setStatus('upload-status', 'error', '❌ 上傳失敗，請重試');
  }

  btn.disabled = false;
  btn.innerHTML = '用 AI 識別並加入衣櫥';
}

// ── 衣櫥 ──────────────────────────────
async function loadWardrobe() {
  try {
    const res = await fetch('/api/wardrobe');
    const items = await res.json();
    const grid = document.getElementById('wardrobe-grid');
    document.getElementById('wardrobe-count').textContent = items.length + ' 件';

    if (items.length === 0) {
      grid.innerHTML = '<div class="wardrobe-empty">衣櫥是空的，快來上傳衣服吧！</div>';
      return;
    }

    grid.innerHTML = items.map(item => `
      <div class="wardrobe-item" id="item-${item.id}">
        <img src="${item.image_url}" alt="${item.clothing_type}" loading="lazy">
        <div class="wardrobe-item-info">
          <div class="wardrobe-item-type">${item.clothing_type || item.category}</div>
          <div class="wardrobe-item-color">${item.color}</div>
          <span class="wardrobe-item-tag">${item.suitable_weather}</span>
        </div>
        <button class="wardrobe-delete" onclick="deleteItem(${item.id})" title="刪除">✕</button>
      </div>
    `).join('');
  } catch (e) {
    console.error('載入衣櫥失敗', e);
  }
}

async function deleteItem(id) {
  if (!confirm('確定要刪除這件衣服嗎？')) return;
  try {
    await fetch(`/api/wardrobe/${id}`, { method: 'DELETE' });
    loadWardrobe();
  } catch (e) {
    alert('刪除失敗');
  }
}

// ── 穿搭建議 ──────────────────────────
async function getSuggestion() {
  const city = document.getElementById('city-input').value.trim() || 'Taipei';
  const btn = document.getElementById('suggest-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>AI 思考中…';
  setStatus('suggest-status', 'loading', '正在根據今日天氣和你的衣櫥生成建議…');
  document.getElementById('suggest-result').style.display = 'none';
  document.getElementById('suggest-placeholder').style.display = 'none';

  try {
    const res = await fetch('/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ city })
    });
    const data = await res.json();

    if (data.error) {
      setStatus('suggest-status', 'error', '❌ ' + data.error);
      document.getElementById('suggest-placeholder').style.display = 'block';
    } else {
      setStatus('suggest-status', '', '');
      const result = document.getElementById('suggest-result');
      result.textContent = data.suggestion;
      result.style.display = 'block';
    }
  } catch (e) {
    setStatus('suggest-status', 'error', '❌ 建議生成失敗，請重試');
    document.getElementById('suggest-placeholder').style.display = 'block';
  }

  btn.disabled = false;
  btn.innerHTML = '重新生成穿搭建議';
}

// ── 工具 ──────────────────────────────
function setStatus(id, type, msg) {
  const el = document.getElementById(id);
  el.className = 'status-msg' + (type ? ' ' + type : '');
  el.textContent = msg;
}

// 初始載入
loadWeather();
loadWardrobe();
</script>
</body>
</html>'''
    return render_template_string(html)

# ========== 錯誤處理 ==========
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '404 Not Found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '500 Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
python app.py
