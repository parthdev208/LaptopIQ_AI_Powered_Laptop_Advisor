import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from datetime import datetime
import requests
from urllib.parse import quote_plus

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))

def fetch_cpu_brand(text):
    if text.startswith('Intel Core i7') or text.startswith('Intel Core i5') or text.startswith('Intel Core i3'):
        return ' '.join(text.split()[0:3])
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


def map_os(os_text):
    if os_text in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif os_text in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


def extract_budget(text):
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)(\s*[kK])?', text)
    if not match:
        return None
    amount = float(match.group(1))
    if match.group(2):
        amount *= 1000
    return int(amount)


def detect_intent(text):
    text = text.lower()
    if 'gaming' in text or 'play' in text or 'fps' in text:
        return 'gaming'
    if 'educ' in text or 'study' in text or 'school' in text or 'college' in text:
        return 'education'
    if 'office' in text or 'work' in text or 'business' in text or 'programming' in text:
        return 'office'
    if 'design' in text or 'photo' in text or 'video' in text or 'edit' in text or 'render' in text:
        return 'creative'
    return 'general'


def build_search_links(company, type_name, ram):
    query = quote_plus(f"{company} {type_name} {ram}GB RAM")
    return {
        'Amazon': f'https://www.amazon.in/s?k={query}',
        'Flipkart': f'https://www.flipkart.com/search?q={query}'
    }


def product_image_url(query):
    # legacy Unsplash 'featured' endpoint is unreliable; return a safe placeholder based on brand
    query = quote_plus(query)
    # simple brand-based placeholders (fallback to a neutral laptop image)
    brand_placeholders = {
        'Apple': 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8',
        'Dell': 'https://images.unsplash.com/photo-1518770660439-4636190af475',
        'HP': 'https://images.unsplash.com/photo-1515879218367-8466d910aaa4',
        'Lenovo': 'https://images.unsplash.com/photo-1527430253228-e93688616381',
        'Asus': 'https://images.unsplash.com/photo-1519389950473-47ba0277781c',
        'Acer': 'https://images.unsplash.com/photo-1541807084-5c52b6b3a2a6'
    }
    for brand, url in brand_placeholders.items():
        if brand.lower() in query.lower():
            return url + '?auto=format&fit=crop&w=400&q=60'
    # default neutral laptop image
    return 'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=400&q=60'


def get_gemini_response(prompt):
    api_key = os.getenv('GEMINI_API_KEY')
    api_url = os.getenv('GEMINI_API_URL')
    if not api_key or not api_url:
        return None
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'prompt': prompt,
            'max_output_tokens': 256,
            'temperature': 0.7
        }
        response = requests.post(api_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'text' in data:
            return data['text']
        if isinstance(data, dict) and 'response' in data and isinstance(data['response'], dict):
            return data['response'].get('output_text') or data['response'].get('text')
        return None
    except Exception:
        return None


def validate_configuration(company, os_name, touchscreen_str, gpu_name=None):
    """Return a list of validation error strings for impossible or uncommon configs."""
    errors = []
    # Apple validations
    if company == 'Apple' and touchscreen_str == 'Yes':
        errors.append('Apple MacBooks do not support touchscreen displays.')
    if company == 'Apple' and os_name != 'Mac':
        errors.append('Apple laptops only use macOS.')
    # GPU compatibility
    if company == 'Apple' and gpu_name is not None and ('Nvidia' in str(gpu_name) or 'GeForce' in str(gpu_name) or 'GTX' in str(gpu_name) or 'RTX' in str(gpu_name)):
        errors.append('Modern Apple laptops do not use Nvidia GPUs.')
    return errors


def get_usage_recommendation(ram, gpu_name):
    try:
        ram = int(ram)
    except Exception:
        ram = 4
    if ram >= 32:
        return "Programming, AI/ML, Video Editing"
    if ram >= 16:
        return "Programming, Video Editing, Gaming"
    if ram >= 8:
        return "Programming, Office Work, College"
    return "Basic Browsing and Office Tasks"


def add_recommendation_to_dataset(rec):
    """Append a recommended item to laptop_data.csv and update in-memory df."""
    global df
    # Build a row with expected columns; fill missing data conservatively
    row = {
        'Company': rec.get('Company', ''),
        'TypeName': rec.get('TypeName', ''),
        'Inches': rec.get('Inches', None),
        'ScreenResolution': rec.get('ScreenResolution', ''),
        'Cpu': rec.get('Cpu', rec.get('Cpu brand', '')),
        'Ram': f"{int(rec.get('Ram',0))}GB" if rec.get('Ram') is not None else '',
        'Memory': rec.get('Memory', ''),
        'Gpu': rec.get('Gpu', rec.get('Gpu brand', '')),
        'OpSys': rec.get('OpSys', rec.get('os', '')),
        'Weight': f"{rec.get('Weight')}kg" if rec.get('Weight') not in (None, '') else '',
        'Price': rec.get('Price', 0)
    }
    try:
        new_df = pd.DataFrame([row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv('laptop_data.csv', index=False)
        try:
            pickle.dump(df, open('df.pkl', 'wb'))
        except Exception:
            # ignore pickle errors
            pass
        return True
    except Exception:
        return False


def recommend_laptops(intent, budget, top_n=3):
    budget = budget or df['Price'].max()
    candidates = df[df['Price'] <= budget].copy()
    if intent == 'gaming':
        candidates = candidates[(candidates['Gpu brand'].str.contains('Nvidia|Radeon|GeForce|RTX|GTX', case=False, na=False)) |
                                 (candidates['TypeName'].str.contains('Gaming', case=False, na=False)) |
                                 (candidates['Ram'] >= 8)]
    elif intent == 'education':
        candidates = candidates[(candidates['TypeName'].str.contains('Notebook|Ultrabook|Student|Business', case=False, na=False)) | (candidates['Weight'] <= 2.0)]
    elif intent == 'office':
        candidates = candidates[(candidates['TypeName'].str.contains('Notebook|Ultrabook|Business', case=False, na=False)) |
                                 (candidates['Ram'] >= 8)]
    elif intent == 'creative':
        candidates = candidates[(candidates['Ram'] >= 8) |
                                 (candidates['Cpu brand'].str.contains('Intel Core i7|Intel Core i5|AMD', case=False, na=False))]
    if candidates.empty:
        candidates = df[df['Price'] <= budget].copy()
    candidates = candidates.sort_values(['Price', 'Ram'], ascending=[False, False])
    candidates = candidates.drop_duplicates(subset=['Company', 'TypeName', 'Ram', 'Cpu', 'Gpu'])
    return candidates.head(top_n)


def format_chat_response(intent, budget, suggestions):
    if not suggestions.empty:
        heading = f"Here are {len(suggestions)} good {intent} laptop options around ₹{budget:,}:"
        lines = [heading]
        for _, row in suggestions.iterrows():
            lines.append(f"- {row['Company']} {row['TypeName']} ({row['Ram']}GB, {row['Cpu']}, {row['Gpu']}) at ₹{int(row['Price']):,}")
        lines.append('I also help you by giving extra shopping links below. You can click the link to compare prices on Amazon or Flipkart.')
        lines.append('If you want, I can refine this further for gaming, study, business, or creative work with a better budget range.')
        return '\n'.join(lines)
    return 'I could not find a matching laptop under that budget, but I can still help you refine the requirement or choose a nearby price range.'


def render_chat_message(role, text):
        # render a chat message with avatar, timestamp and nice bubbles
        ts = datetime.now().strftime('%H:%M')
        safe_text = text.replace('\n', '<br/>')
        if role == 'user':
                html = f"""
                <div style='display:flex;align-items:flex-end;justify-content:flex-end'>
                    <div style='max-width:78%;'>
                        <div class='chat-bubble user'>
                            {safe_text}
                            <div style='font-size:10px; opacity:0.7; text-align:right;margin-top:6px'>{ts}</div>
                        </div>
                    </div>
                    <div style='width:42px; height:42px; margin-left:8px;'>
                        <img src='https://api.dicebear.com/6.x/initials/svg?seed=You' width='42' height='42' style='border-radius:50%'/>
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
                html = f"""
                <div style='display:flex;align-items:flex-start;justify-content:flex-start'>
                    <div style='width:42px; height:42px; margin-right:8px;'>
                        <img src='https://api.dicebear.com/6.x/bottts/svg?seed=LaptopAI' width='42' height='42' style='border-radius:50%'/>
                    </div>
                    <div style='max-width:78%;'>
                        <div class='chat-bubble assistant'>
                            {safe_text}
                            <div style='font-size:10px; opacity:0.7; text-align:left;margin-top:6px'>{ts}</div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)


df = pd.read_csv('laptop_data.csv')
df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')
df['Cpu brand'] = df['Cpu'].apply(fetch_cpu_brand)
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df['os'] = df['OpSys'].apply(map_os)

# Title moved to header banner above; do not render the plain old page title.
# st.title("Laptop Predictor")

st.markdown(
        """
        <style>
        /* Page background gradient */
        body {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #111827 100%);
                color: #e6eef8;
        }

        /* Header banner */
        .header-banner {display:flex;align-items:center;gap:18px;padding:20px;border-radius:16px;background:linear-gradient(90deg,#3b82f6,#8b5cf6);color:#fff;margin-bottom:18px}
        .header-banner img {height:64px;border-radius:12px}
        .header-title {font-size:22px;font-weight:700;margin:0}
        .header-sub {opacity:0.95;margin:0;font-size:13px}

        /* Card styles */
        .card {background:#1e293b;border:1px solid #334155;border-radius:16px;padding:14px;color:#e6eef8}

        /* Chat bubbles */
        .chat-bubble {border-radius: 14px; padding: 12px 14px; margin: 8px 0; display:inline-block; max-width:78%; line-height:1.4}
        .chat-bubble.user {background:#0b1220; color:#e6eef8; margin-left:auto}
        .chat-bubble.assistant {background:#ffffff; color:#0b1220; margin-right:auto; box-shadow: 0 2px 6px rgba(16,24,40,0.06)}

        /* Metric colors (streamlit metrics will inherit colors) */
        .stMetric {color:#e6eef8}

        /* Small helper for HTML buttons used in cards */
        .action-btn {font-weight:700;padding:10px 18px;border-radius:8px;border:none;cursor:pointer}

        /* Top buttons styling (JS fallback also present) */
        .stButton>button{border-radius:10px;padding:10px 16px;font-weight:700}
        .stButton:nth-of-type(1) > button{background:#f59e0b;color:#fff}
        .stButton:nth-of-type(2) > button{background:#8b5cf6;color:#fff}
        </style>

        <!-- Header banner markup -->
        <div class="header-banner">
            <img src="https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=140&q=60" alt="laptop" />
            <div>
                <div class="header-title">🧠 LaptopIQ — AI Powered Laptop Advisor</div>
                <div class="header-sub">Smart price predictions, practical recommendations, and shopping help</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latest_recommendations' not in st.session_state:
    st.session_state.latest_recommendations = []
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ''
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type_name = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop', min_value=0.5, max_value=10.0, value=2.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])

# IPS
ips = st.selectbox('IPS', ['No','Yes'])

# Screen size
screen_size = st.number_input('Screen Size (inches)', min_value=8.0, max_value=20.0, value=15.6, step=0.1, format='%.1f')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# storage
hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

selected_os = st.selectbox('OS', df['os'].unique())

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button('💰 Predict Price', use_container_width=True)
with col2:
    advisor_btn = st.button('🚀 Smart Laptop Advisor', use_container_width=True)

if advisor_btn:
    st.session_state.chat_open = True

# Style the two top buttons (attempt JS-based coloring for clarity). If JS blocked, default styling applies.
st.markdown("""
<script>
try{
  const buttons = Array.from(document.querySelectorAll('button'))
  buttons.forEach(b=>{
    const text = (b.innerText||'').trim()
    if(text.startsWith('💰')){ b.style.background='#f59e0b'; b.style.color='white'; b.style.border='none'; b.style.fontWeight='700' }
    if(text.startsWith('🚀')){ b.style.background='#8b5cf6'; b.style.color='white'; b.style.border='none'; b.style.fontWeight='700' }
  })
}catch(e){/* ignore */}
</script>
""", unsafe_allow_html=True)

if predict_btn:
    # query
    ppi = None
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    if screen_size <= 0:
        st.error('Screen Size must be greater than zero.')
    else:
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
        query = pd.DataFrame([[
            company,
            type_name,
            ram,
            weight,
            touchscreen_val,
            ips_val,
            ppi,
            cpu,
            hdd,
            ssd,
            gpu,
            selected_os
        ]], columns=[
            'Company',
            'TypeName',
            'Ram',
            'Weight',
            'Touchscreen',
            'IPS',
            'ppi',
            'Cpu brand',
            'HDD',
            'SSD',
            'Gpu brand',
            'os'
        ])

        # compatibility warnings
        if company == 'Apple' and ('Nvidia' in str(gpu) or 'GeForce' in str(gpu) or 'GTX' in str(gpu) or 'RTX' in str(gpu)):
            st.warning('Modern Apple laptops do not use Nvidia GPUs.')

        # validate configuration
        validation_errors = validate_configuration(company, selected_os, 'Yes' if touchscreen_val == 1 else 'No', gpu)

        # make prediction (still compute numeric estimate)
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        # Availability score
        availability_score = 100
        if company == 'Apple' and touchscreen_val == 1:
            availability_score = 0
        if company == 'Apple' and selected_os != 'Mac':
            availability_score = 0

        # Confidence (simple heuristic / placeholder)
        confidence = 92

        # Price range
        low = max(0, predicted_price - 5000)
        high = predicted_price + 5000

        # Show key metrics (Price / Availability / Confidence)
        m1, m2, m3 = st.columns(3)
        m1.metric('💰 Price', f'₹{predicted_price:,}')
        m2.metric('✅ Available', 'Yes' if availability_score > 0 else 'No')
        m3.metric('🎯 Confidence', f'{confidence}%')
        st.progress(confidence / 100)

        # If invalid, show errors and suggested real laptop
        if validation_errors:
            st.error('⚠️ This laptop configuration is not available in the real world.')
            for err in validation_errors:
                st.warning(err)


            st.subheader('Suggested Real Laptop')
            suggested_ram = min(max(ram, 8), 16)
            suggested_ssd = ssd if ssd and ssd > 0 else 256
            st.success(f"""
Apple MacBook Air

• {suggested_ram}GB RAM
• {suggested_ssd}GB SSD
• macOS
• Non-Touchscreen

Estimated Price: ₹{predicted_price:,}
""")
            st.info('Reason: configuration not available (Apple touchscreen / OS mismatch / GPU mismatch).')

        else:
            st.success(f'Predicted Price: ₹{predicted_price:,}')
            st.success(f'Expected Market Price Range: ₹{low:,} - ₹{high:,}')

            # Similar real laptops (closest by price)
            similar = df.copy()
            similar['price_diff'] = (similar['Price'] - predicted_price).abs()
            similar = similar.sort_values('price_diff').head(3)
            if not similar.empty:
                st.subheader('Similar Real Laptops')
                st.dataframe(similar[['Company','TypeName','Ram','Cpu','Price']].reset_index(drop=True))

            # Feature score card (simple heuristics)
            gaming = 40 + (ram/64)*60
            if any([k in str(cpu) for k in ['i7','Ryzen 7','Radeon','RTX','GeForce']]):
                gaming += 15
            programming = 60 + (ram/32)*30
            office = 80 + (ram/64)*20
            video = 40 + (ram/64)*40
            battery = 70 if weight <= 2.0 else 60

            st.subheader('Laptop Analysis')
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Gaming', f'{int(min(100,gaming))}/100')
            c2.metric('Programming', f'{int(min(100,programming))}/100')
            c3.metric('Office Work', f'{int(min(100,office))}/100')
            c4.metric('Video Editing', f'{int(min(100,video))}/100')
            c5.metric('Battery Life', f'{int(min(100,battery))}/100')

            # Market compare
            market_avg = int(df['Price'].mean())
            if predicted_price > market_avg:
                st.info('Above average market price')
            else:
                st.info('Below average market price')

if st.session_state.chat_open:
    chat_input = st.text_area('Type your question here', value=st.session_state.chat_input, height=120, key='chat_input_widget')
    send_col = st.columns([1])[0]
    if send_col.button('Send Query'):
        if chat_input.strip():
            st.session_state.chat_history.append({'role': 'user', 'text': chat_input.strip()})
            intent = detect_intent(chat_input)
            budget = extract_budget(chat_input)
            recommendations = recommend_laptops(intent, budget)
            response_text = None
            gemini_text = get_gemini_response(chat_input)
            if gemini_text:
                response_text = gemini_text
            if not response_text:
                response_text = format_chat_response(intent, budget or int(df['Price'].max()), recommendations)
            st.session_state.chat_history.append({'role': 'assistant', 'text': response_text})
            st.session_state.latest_recommendations = recommendations.to_dict(orient='records')
            st.session_state.chat_input = ''

    for message in st.session_state.chat_history:
        render_chat_message(message['role'], message['text'])

if st.session_state.latest_recommendations:
        st.subheader('Recommended laptops')
        for rec in st.session_state.latest_recommendations:
                links = build_search_links(rec['Company'], rec['TypeName'], rec['Ram'])
                # prefer dataset ImageURL if present
                image_url = rec.get('ImageURL') or product_image_url(f"{rec['Company']} {rec['TypeName']} laptop")

                # availability
                touch_val = rec.get('Touchscreen', '')
                touch_str = str(touch_val).lower()
                touch_enabled = touch_str in ('1', 'true', 'yes')
                availability = '✅ Available'
                if rec.get('Company') == 'Apple' and touch_enabled:
                        availability = '❌ Not Available'

                price_val = int(rec.get('Price', 0) or 0)
                low_p = int(price_val * 0.95)
                high_p = int(price_val * 1.05)

                usage = get_usage_recommendation(rec.get('Ram', 4), rec.get('Gpu', ''))

                card_html = f"""
<div style="padding:16px;border-radius:12px;background:#0f1724;color:#fff;margin-bottom:18px;border:1px solid rgba(255,255,255,0.04);display:flex;gap:16px;align-items:flex-start">
    <div style="flex:0 0 260px;">
        <img src="{image_url}" width="250" style="border-radius:8px;object-fit:cover;"/>
    </div>
    <div style="flex:1">
        <h3 style="margin:0 0 6px 0;color:#fff">🖥 {rec.get('Company','')} {rec.get('TypeName','')}</h3>
        <div style="margin-bottom:8px;color:#cbd5e1">💰 Predicted Price: ₹{price_val:,} &nbsp; | &nbsp; Range: ₹{low_p:,} - ₹{high_p:,}</div>
        <div style="margin-bottom:8px">🔎 <b>Availability:</b> {availability} &nbsp; • &nbsp; <b>Best For:</b> {usage}</div>
        <div style="margin-bottom:8px">⚙ <b>Specs:</b> RAM: {rec.get('Ram')}GB | CPU: {rec.get('Cpu')} | GPU: {rec.get('Gpu')} | OS: {rec.get('OpSys')}</div>
        <div style="margin-top:12px">
            <a href="{links['Amazon']}" target="_blank"><button style="background:#22c55e;color:white;border:none;padding:10px 18px;border-radius:8px;font-weight:bold;cursor:pointer;margin-right:10px">🛒 Buy Now</button></a>
            <a href="{links['Flipkart']}" target="_blank"><button style="background:#3b82f6;color:white;border:none;padding:10px 18px;border-radius:8px;font-weight:bold;cursor:pointer">🔍 View Deals</button></a>
        </div>
    </div>
</div>
"""

                try:
                    # Render the full HTML card (including action buttons) in one safe block
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception:
                    # fallback to simple layout with HTML buttons
                    st.image(image_url, width=250)
                    st.markdown(f"**{rec.get('Company','')} {rec.get('TypeName','')}**")
                    st.markdown(f"Price: ₹{price_val:,}")
                    st.markdown(f"Availability: {availability}")
                    buttons_html = f"""
<a href="{links['Amazon']}" target="_blank"><button style="background:#22c55e;color:white;border:none;padding:10px 18px;border-radius:8px;font-weight:bold;cursor:pointer;margin-right:10px">🛒 Buy Now</button></a>
<a href="{links['Flipkart']}" target="_blank"><button style="background:#3b82f6;color:white;border:none;padding:10px 18px;border-radius:8px;font-weight:bold;cursor:pointer">🔍 View Deals</button></a>
"""
                    st.markdown(buttons_html, unsafe_allow_html=True)
                    st.write('')

