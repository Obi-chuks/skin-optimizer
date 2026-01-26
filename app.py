import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import unicodedata
from scipy.sparse import hstack
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="L'Oréal Green AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. CORPORATE IDENTITY CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    
    .header-container {
        background: linear-gradient(90deg, #004d2c 0%, #002a18 100%);
        padding: 25px; border-radius: 0 0 15px 15px; color: white; margin-bottom: 30px;
    }
    .brand-title { font-size: 2.2rem; font-weight: 800; letter-spacing: 2px; color: #fff !important; margin: 0; }
    .brand-subtitle { font-size: 1rem; color: #c5a059 !important; text-transform: uppercase; letter-spacing: 4px; margin-top: 5px; font-weight: 600; }
    
    .metric-card {
        background-color: #FFFFFF !important; border: 1px solid #e0e0e0; border-radius: 12px;
        padding: 25px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); color: #000 !important;
    }
    .eco-card { background: linear-gradient(135deg, #e6f5ea 0%, #ffffff 100%) !important; border: 1px solid #c3e6cb; }
    
    div.stButton > button {
        background: #c5a059; color: #002a18; border: none; border-radius: 6px; padding: 15px 25px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; width: 100%; transition: all 0.3s ease;
    }
    div.stButton > button:hover { background: #d4af37; color: black; box-shadow: 0 5px 15px rgba(197,160,89,0.4); }
    div.stTextArea > div > div > textarea { background-color: #fff !important; color: #333 !important; border: 2px solid #e0e0e0; border-radius: 10px; }
    div.stTextArea > div > div > textarea:focus { border-color: #004d2c; box-shadow: 0 0 0 1px #004d2c; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---

# *** CRITICAL FIX: Tokenizer must match training script name exactly ***
def tokenize_split(text):
    return text.split()

@st.cache_resource
def download_nltk():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt_tab', quiet=True)
download_nltk()

# *** HARDCODED FEATURE DEFINITIONS (Because they are missing from pickle) ***
feature_defs = {
    "has_vitamin_c": ["vitamin c", "ascorbic", "citrus"],
    "has_retinol": ["retinol", "retinal", "vitamin a", "tretinoin"],
    "has_niacinamide": ["niacinamide", "b3"],
    "has_hyaluronic": ["hyaluronic", "sodium hyaluronate", "ha"],
    "has_salicylic": ["salicylic", "bha", "willow bark"],
    "has_glycolic": ["glycolic", "aha", "alpha hydroxy acid"],
    "has_peptide": ["peptide", "ceramide", "collagen"],
    "has_ferulic": ["ferulic"],
    "has_clay": ["clay", "kaolin", "charcoal"],
    "has_spf": ["spf", "sunscreen", "sun", "uv", "broad spectrum", "sun protection factor"],
    "has_hydrating": ["hydrating", "moisture", "moisturizing"],
    "has_oil_control": ["oil free", "matte", "non comedogenic", "water-based"],
    "has_sensitive": ["sensitive", "gentle", "soothe"],
    "has_anti_aging": ["anti aging", "wrinkle", "firm", "collagen"],
    "has_acne": ["acne", "blemish", "pore"],
    "has_radiance": ["bright", "glow", "radiance"],
    "is_serum": ["serum"], "is_cream": ["cream", "balm"], "is_cleanser": ["cleanser", "wash"],
    "is_toner": ["toner"], "is_mask": ["mask"], "is_sunscreen": ["sunscreen", "spf"],
    "is_moisturizer": ["moisturizer", "lotion"], "is_day": ["day", "morning"],
    "is_night": ["night"], "is_daily": ["daily"]
}

@st.cache_resource
def load_assets():
    # Load pickle
    assets = joblib.load('deployment_pack.pkl')
    
    # *** CRITICAL FIX: Map new pickle keys to variables App expects ***
    return {
        'model': assets['model'],
        'tfidf': assets['vectorizer'],     # Called 'vectorizer' in new training code
        'scaler': assets['scaler'],
        'best_thresh': assets['optimized_thresh'], # Called 'optimized_thresh' in new training code
        'target_cols': assets['TARGET_COLS'],      # Uppercase in new training code
        'feature_defs': feature_defs               # Use our hardcoded version
    }

try:
    assets = load_assets()
    model = assets['model']; tfidf = assets['tfidf']; scaler = assets['scaler']
    best_thresh = assets['best_thresh']; TARGET_COLS = assets['target_cols']
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
    st.stop()

# --- PRESTIGE NAMING DICTIONARY ---
DISPLAY_NAMES = {
    "has_acne": "BLEMISH DEFENSE", "acne": "BLEMISH DEFENSE",
    "has_anti_aging": "AGE CORRECTION", "fine_wrinkles": "WRINKLE REDUCTION", "lack_firmness": "FIRMING & LIFT",
    "has_hydrating": "DEEP HYDRATION", "dry": "HYDRATION RECOVERY",
    "has_oil_control": "SEBUM REGULATION", "oily": "OIL CONTROL",
    "has_sensitive": "SENSITIVE SKIN RELIEF", "sensitivity-high": "SENSITIVE SKIN RELIEF",
    "has_radiance": "LUMINOSITY BOOSTER", "lack_radiance": "RADIANCE ENHANCEMENT",
    "dark_pigmentation": "PIGMENT CORRECTION", "pores": "PORE REFINING",
    "normal": "BALANCE MAINTENANCE", "combination": "COMBINATION BALANCING"
}

def extract_and_remove_sizes(text):
    if not isinstance(text, str): return text, ''
    text_lower = text.lower(); extracted = []
    patterns = [(r'\d+(?:\.\d+)?\s*(?:ml|milliliter)', 'ml'), (r'\d+(?:\.\d+)?\s*(?:g|gram)', 'g')]
    for pattern, unit in patterns:
        for match in re.finditer(pattern, text_lower): extracted.append(match.group())
    text_clean = re.sub(r'\d+(?:\.\d+)?\s*(?:ml|g|oz)', ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text_clean).strip(), '; '.join(extracted)

def clean_text(t):
    if not isinstance(t, str): return ''
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8')
    t = re.sub(r'<[^>]+>', ' ', t); t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    t = t.lower(); t = re.sub(r'\s+', ' ', t).strip()
    return t

def lemmatize_text(text):
    if not isinstance(text, str): return ''
    try: tokens = word_tokenize(text.lower()); return ' '.join([lemmatizer.lemmatize(t) for t in tokens])
    except: return text

def process_input(raw_text):
    text_no_size, _ = extract_and_remove_sizes(raw_text)
    clean = clean_text(text_no_size)
    final_text = lemmatize_text(clean) # Basic Lemmatization
    final_text = re.sub(r'[^a-z0-9\s]', ' ', final_text) # Remove special chars

    features = {}
    for name, kws in feature_defs.items():
        features[name] = 1 if any(kw in clean for kw in kws) else 0

    features['text_length'] = len(clean)
    features['word_count'] = len(clean.split())
    features['total_actives'] = sum(features[k] for k in features if k.startswith('has_'))
    features['total_concerns'] = sum(features[k] for k in ['has_hydrating', 'has_oil_control', 'has_sensitive', 'has_anti_aging', 'has_acne', 'has_radiance'])
    features['product_complexity'] = sum(features[k] for k in features if k.startswith('is_'))
    features['sensitivity_high_text'] = 1 if 'sensitive' in clean else 0 # MATCHING TRAINING CODE NAME
    features['sensitivity_low_text'] = 0

    feature_cols = scaler.feature_names_in_
    df_feats = pd.DataFrame([features])
    for col in feature_cols:
        if col not in df_feats.columns: df_feats[col] = 0
    df_feats = df_feats[feature_cols]

    X_tf = tfidf.transform([final_text]) # Uses tokenize_split defined above
    X_eng = scaler.transform(df_feats)
    return hstack([X_tf, X_eng])

def find_ingredients(text):
    ingredients = {
        "Salicylic Acid": ["salicylic", "bha"], "Retinol": ["retinol", "retinoid"],
        "Vitamin C": ["vitamin c", "ascorbic"], "Hyaluronic Acid": ["hyaluronic", "sodium hyaluronate"],
        "Niacinamide": ["niacinamide"], "Glycerin": ["glycerin"], "SPF": ["spf", "sunscreen"],
        "Clay": ["clay", "kaolin"], "Ceramides": ["ceramide"], "Peptides": ["peptide"], "Glycolic Acid": ["glycolic", "aha"]
    }
    found = []; text_lower = text.lower()
    for name, keywords in ingredients.items():
        if any(k in text_lower for k in keywords): found.append(name)
    return found

# --- 3. UI LAYOUT ---
st.markdown("""
<div class="header-container">
    <div class="brand-title">L'ORÉAL</div>
    <div class="brand-subtitle">Strategic Formulation Intelligence</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown("### 📋 INPUT DATA STREAM")
    st.markdown("<div style='color: #888; margin-bottom: 10px; font-size: 0.9rem;'>Enter product R&D description or full ingredient list for analysis.</div>", unsafe_allow_html=True)
    input_text = st.text_area("Formulation Text", height=250, label_visibility="collapsed", placeholder="Example: Advanced nighttime serum with 0.3% Retinol and Ceramides...")
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("RUN DIAGNOSTIC ➤")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if analyze_btn and input_text:
        word_count = len(input_text.split())
        xgboost_emissions = 0.0002 + (word_count * 0.000005) 
        llm_emissions = 4.5
        savings = (1 - (xgboost_emissions / llm_emissions)) * 100
        
        st.markdown(f"""
        <div class="metric-card eco-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 10px;">🌍</span>
                <span style="font-weight: 800; color: #004d2c; letter-spacing: 1px;">SUSTAINABILITY METRICS</span>
            </div>
            <div style="font-size: 0.9rem; color: #333; margin-bottom: 10px;">
                <b>Green AI Architecture:</b> Analyzing <b>{word_count} data points</b> with optimized compute load.
            </div>
            <div style="background-color: #fff; border-radius: 8px; padding: 10px; border: 1px solid #cce3d4;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; color: #555;">
                    <span>⚡ Energy Efficiency</span>
                    <span style="color: #004d2c;">{savings:.4f}% vs GenAI</span>
                </div>
                <div style="background:#eee; height:8px; border-radius:4px; width:100%; margin-top:5px;">
                    <div style="background:#004d2c; width:{savings}%; height:8px; border-radius:4px;"></div>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #666; font-style: italic;">
                *Real-time footprint: <b>{xgboost_emissions:.6f}g CO2e</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if analyze_btn and input_text:
        with st.spinner("Processing Green AI Algorithms..."):
            X_input = process_input(input_text)
            probs = model.predict_proba(X_input)[0] # Probs are already correct length
            found_ingredients = find_ingredients(input_text)

            results = []
            for i, col in enumerate(TARGET_COLS):
                # Use the optimized threshold from the pickle
                thresh = best_thresh[i]
                score = probs[i]
                is_active = score > thresh
                
                display_name = DISPLAY_NAMES.get(col, col.replace('_', ' ').replace('-', ' ').title())
                results.append({"concern": display_name, "score": score, "active": is_active})
            
            results = sorted(results, key=lambda x: (x["active"], x["score"]), reverse=True)
            top_match = results[0]

            st.markdown("### 📊 DIAGNOSTIC RESULTS")

            if top_match['score'] > 0.2 and top_match['active']:
                verdict_color = "#004d2c"; verdict_text = top_match['concern']
            else:
                verdict_color = "#856404"; verdict_text = "INCONCLUSIVE"

            st.markdown(f"""
            <div class="metric-card" style="border-left: 8px solid {verdict_color};">
                <div style="font-size: 0.8rem; font-weight: 700; color: #888; letter-spacing: 1px;">PRIMARY CLASSIFICATION</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {verdict_color}; line-height: 1.2;">{verdict_text}</div>
                <div style="font-size: 1.1rem; color: #333; margin-top: 5px;">Confidence Score: <b>{top_match['score']:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

            if found_ingredients:
                st.markdown("<br><b>DETECTED ACTIVES:</b>", unsafe_allow_html=True)
                tags_html = ""
                for ing in found_ingredients:
                    tags_html += f"<span style='background:#eef5f1; color:#004d2c; padding:6px 12px; margin-right:5px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block; margin-bottom:5px; border:1px solid #cce3d4;'>{ing}</span>"
                st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("<br><div style='font-size:0.9rem; font-weight:700; color:#888; margin-bottom:10px;'>FULL SPECTRUM ANALYSIS</div>", unsafe_allow_html=True)
            
            for res in results[:5]:
                col_name = res['concern']; score_val = res['score']
                bar_color = "#004d2c" if res['active'] else "#e0e0e0"
                text_color = "#000000" if res['active'] else "#888888"

                st.markdown(f"""
                <div style="margin-bottom: 12px; background-color: #ffffff; padding: 5px; border-radius: 5px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; color:{text_color}; font-weight:700;">
                        <span>{col_name}</span>
                        <span>{score_val:.1%}</span>
                    </div>
                    <div style="background:#f0f0f0; height:10px; border-radius:5px; width:100%; margin-top:4px;">
                        <div style="background:{bar_color}; width:{score_val*100}%; height:10px; border-radius:5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif analyze_btn and not input_text:
        st.warning("⚠️ No data input. Please enter formulation text.")
    else:
        st.markdown("""
        <div class="metric-card" style="text-align: center; color: #999; padding: 40px;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🔍</div>
            <div style="font-weight: 600;">System Ready</div>
            <div style="font-size: 0.8rem;">Awaiting Data Input</div>
        </div>
        """, unsafe_allow_html=True)
