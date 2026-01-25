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

# --- PAGE CONFIGURATION (Wide Mode for Dashboard Feel) ---
st.set_page_config(
    page_title="L'Oréal Green AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. CORPORATE IDENTITY CSS (The Visual Overhaul) ---
st.markdown("""
<style>
    /* IMPORT FONT (Roboto/Helvetica for clean business look) */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        background-color: #F8F9FA; /* Light Grey Background */
    }

    /* HEADER STYLE */
    .header-container {
        background: linear-gradient(90deg, #004d2c 0%, #002a18 100%);
        padding: 25px;
        border-radius: 0px 0px 15px 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #ffffff;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 1rem;
        color: #c5a059; /* Gold */
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-top: 5px;
    }

    /* CARD DESIGN */
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* BUTTON STYLING */
    div.stButton > button {
        background: #c5a059; /* Gold Background */
        color: #002a18; /* Dark Green Text */
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(197, 160, 89, 0.3);
    }
    div.stButton > button:hover {
        background: #d4af37;
        color: black;
    }

    /* TEXT INPUT STYLING */
    div.stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        color: #333;
    }
    div.stTextArea > div > div > textarea:focus {
        border-color: #004d2c;
        box-shadow: 0 0 0 1px #004d2c;
    }
    
    /* PROGRESS BAR OVERRIDE */
    .stProgress > div > div > div > div {
        background-color: #004d2c; /* Corporate Green Bars */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (Unchanged) ---
def simple_tokenizer(text):
    return text.split()

@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
download_nltk()

@st.cache_resource
def load_assets():
    return joblib.load('deployment_pack.pkl')

try:
    assets = load_assets()
    model = assets['model']
    tfidf = assets['tfidf']
    scaler = assets['scaler']
    feature_defs = assets['feature_defs']
    best_thresh = assets['best_thresh']
    TARGET_COLS = assets['target_cols']
    lemmatizer = WordNetLemmatizer()
except FileNotFoundError:
    st.error("⚠️ SYSTEM ERROR: deployment_pack.pkl missing.")
    st.stop()

def extract_and_remove_sizes(text):
    if not isinstance(text, str): return text, ''
    text_lower = text.lower()
    extracted = []
    patterns = [(r'\d+(?:\.\d+)?\s*(?:ml|milliliter)', 'ml'), (r'\d+(?:\.\d+)?\s*(?:g|gram)', 'g')]
    for pattern, unit in patterns:
        for match in re.finditer(pattern, text_lower):
            extracted.append(match.group())
    text_clean = re.sub(r'\d+(?:\.\d+)?\s*(?:ml|g|oz)', ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text_clean).strip(), '; '.join(extracted)

def clean_text(t):
    if not isinstance(t, str): return ''
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8')
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    t = t.lower()
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def lemmatize_text(text):
    if not isinstance(text, str): return ''
    try:
        tokens = word_tokenize(text.lower())
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])
    except:
        return text

def process_input(raw_text):
    text_no_size, _ = extract_and_remove_sizes(raw_text)
    clean = clean_text(text_no_size)
    final_text = lemmatize_text(clean)
    final_text = re.sub(r'[^a-z0-9\s]', ' ', final_text)

    features = {}
    for name, kws in feature_defs.items():
        features[name] = 1 if any(kw in clean for kw in kws) else 0

    features['text_length'] = len(clean)
    features['word_count'] = len(clean.split())
    features['total_actives'] = sum(features[k] for k in features if k.startswith('has_'))
    features['total_concerns'] = sum(features[k] for k in ['has_hydrating', 'has_oil_control', 'has_sensitive', 'has_anti_aging', 'has_acne', 'has_radiance'])
    features['product_complexity'] = sum(features[k] for k in features if k.startswith('is_'))
    features['sensitivity_high'] = 1 if 'sensitive' in clean else 0
    features['sensitivity_low'] = 0

    feature_cols = scaler.feature_names_in_
    df_feats = pd.DataFrame([features])
    for col in feature_cols:
        if col not in df_feats.columns:
            df_feats[col] = 0
    df_feats = df_feats[feature_cols]

    X_tf = tfidf.transform([final_text])
    X_eng = scaler.transform(df_feats)
    return hstack([X_tf, X_eng])

def find_ingredients(text):
    ingredients = {
        "Salicylic Acid": ["salicylic", "bha"],
        "Retinol": ["retinol", "retinoid"],
        "Vitamin C": ["vitamin c", "ascorbic"],
        "Hyaluronic Acid": ["hyaluronic", "sodium hyaluronate"],
        "Niacinamide": ["niacinamide"],
        "Glycerin": ["glycerin"],
        "SPF": ["spf", "sunscreen"],
        "Clay": ["clay", "kaolin", "charcoal"],
        "Ceramides": ["ceramide"],
        "Peptides": ["peptide"]
    }
    found = []
    text_lower = text.lower()
    for name, keywords in ingredients.items():
        if any(k in text_lower for k in keywords):
            found.append(name)
    return found

# --- 3. UI LAYOUT: THE EXECUTIVE DASHBOARD ---

# HEADER
st.markdown("""
<div class="header-container">
    <div class="brand-title">L'ORÉAL</div>
    <div class="brand-subtitle">Strategic Formulation Intelligence</div>
</div>
""", unsafe_allow_html=True)

# MAIN LAYOUT: 2 Columns (Input Left, Output Right)
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown("### 📋 INPUT DATA STREAM")
    st.markdown("<div style='color: #666; margin-bottom: 10px; font-size: 0.9rem;'>Enter product R&D description or full ingredient list for analysis.</div>", unsafe_allow_html=True)
    
    input_text = st.text_area("Formulation Text", height=250, label_visibility="collapsed", placeholder="Example: Advanced nighttime serum with 0.3% Retinol and Ceramides...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("RUN DIAGNOSTIC ➤")

with col_right:
    if analyze_btn and input_text:
        with st.spinner("Processing Green AI Algorithms..."):
            # PROCESS
            X_input = process_input(input_text)
            probs = model.predict_proba(X_input)[0]
            found_ingredients = find_ingredients(input_text)

            # SORT
            results = []
            for i, col in enumerate(TARGET_COLS):
                results.append({"concern": col, "score": probs[i], "active": probs[i] > best_thresh[i]})
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            top_match = results[0]

            # --- THE RESULTS DASHBOARD ---
            st.markdown("### 📊 DIAGNOSTIC RESULTS")

            # 1. TOP CARD (The Verdict)
            if top_match['score'] > 0.4:
                verdict_color = "#004d2c" # Green
                verdict_text = top_match['concern'].upper()
            else:
                verdict_color = "#856404" # Warning Gold
                verdict_text = "INCONCLUSIVE"

            st.markdown(f"""
            <div class="metric-card" style="border-left: 8px solid {verdict_color}; background: #fff;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #999; letter-spacing: 1px;">PRIMARY CLASSIFICATION</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {verdict_color}; line-height: 1.2;">{verdict_text}</div>
                <div style="font-size: 1.1rem; color: #333; margin-top: 5px;">Confidence Score: <b>{top_match['score']:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

            # 2. INGREDIENT TAGS
            if found_ingredients:
                st.markdown("<br><b>DETECTED ACTIVES:</b>", unsafe_allow_html=True)
                tags_html = ""
                for ing in found_ingredients:
                    tags_html += f"<span style='background:#eef5f1; color:#004d2c; padding:5px 10px; margin-right:5px; border-radius:15px; font-size:0.85rem; font-weight:600; display:inline-block; margin-bottom:5px;'>{ing}</span>"
                st.markdown(tags_html, unsafe_allow_html=True)

            # 3. DETAILED METRICS GRID
            st.markdown("<br><div style='font-size:0.9rem; font-weight:700; color:#999; margin-bottom:10px;'>FULL SPECTRUM ANALYSIS</div>", unsafe_allow_html=True)
            
            # Show top 5 results as nice bars
            for res in results[:5]:
                col_name = res['concern'].replace('_', ' ').title()
                score_val = res['score']
                
                # Logic for bar color: Green if active, Gray if inactive
                bar_color = "#004d2c" if res['active'] else "#e0e0e0"
                text_color = "#000" if res['active'] else "#999"

                st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; color:{text_color}; font-weight:600;">
                        <span>{col_name}</span>
                        <span>{score_val:.1%}</span>
                    </div>
                    <div style="background:#f0f0f0; height:8px; border-radius:4px; width:100%;">
                        <div style="background:{bar_color}; width:{score_val*100}%; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif analyze_btn and not input_text:
        st.warning("⚠️ No data input. Please enter formulation text.")
    else:
        # Placeholder state before analysis
        st.markdown("""
        <div class="metric-card" style="text-align: center; color: #999; padding: 40px;">
            <div style="font-size: 3rem;">🔍</div>
            <div style="margin-top: 10px;">Awaiting Data Input</div>
            <div style="font-size: 0.8rem;">System Ready</div>
        </div>
        """, unsafe_allow_html=True)
