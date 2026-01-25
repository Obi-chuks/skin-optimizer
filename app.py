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
    page_title="L'Oréal Skin AI",
    page_icon="🌿",
    layout="centered"
)

# --- CUSTOM CSS: THE "CLASSIC" LOOK ---
st.markdown("""
<style>
    /* 1. Main Background & Text */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 2. Header Styling */
    h1 {
        color: #004d2c; /* Deep L'Oreal Green */
        font-family: 'Georgia', serif; /* Classic Serif Font */
        font-weight: 300;
        text-align: center;
        border-bottom: 2px solid #c5a059; /* Muted Gold */
        padding-bottom: 15px;
    }
    h3 {
        color: #333333;
        text-align: center;
        font-weight: 300;
        font-size: 1.2rem;
    }

    /* 3. The "Analyze" Button - Green & Gold */
    div.stButton > button {
        background-color: #004d2c;
        color: #c5a059; /* Gold Text */
        border: 1px solid #c5a059;
        border-radius: 0px; /* Sharp, classic edges */
        padding: 10px 25px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00331d;
        color: #e0c17b;
        border-color: #e0c17b;
    }

    /* 4. Result Cards */
    .result-card {
        background-color: #FAFAFA; /* Very light grey/white */
        border-left: 5px solid #c5a059; /* Gold accent on left */
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-title {
        color: #004d2c;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-score {
        color: #555;
        font-size: 0.9rem;
    }
    .ingredient-box {
        background-color: #eef5f1; /* Pale Green */
        color: #004d2c;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid #cce3d4;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DEFINE TOKENIZER (CRITICAL) ---
def simple_tokenizer(text):
    return text.split()

# --- LOAD RESOURCES ---
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
    st.error("⚠️ deployment_pack.pkl not found! Please upload it to GitHub.")
    st.stop()

# --- PROCESSING FUNCTIONS ---
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

# --- UI LAYOUT ---
# 1. Header
st.markdown("<h1>L'ORÉAL SKINCARE OPTIMIZER</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-Powered Formulation Analysis</h3>", unsafe_allow_html=True)
st.divider()

# 2. Input
input_text = st.text_area("", height=150, placeholder="Paste product description or ingredient list here...")

# 3. Action Button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_btn = st.button("ANALYZE FORMULATION")

# 4. Results Section
if analyze_btn:
    if input_text:
        with st.spinner("Analyzing chemical composition..."):
            # A. Process Data
            X_input = process_input(input_text)
            probs = model.predict_proba(X_input)[0]
            found_ingredients = find_ingredients(input_text)

            # B. Sort Data
            results = []
            for i, col in enumerate(TARGET_COLS):
                results.append({"concern": col, "score": probs[i], "active": probs[i] > best_thresh[i]})
            
            # Sort: Active first, then by score
            results = sorted(results, key=lambda x: (x["active"], x["score"]), reverse=True)

            st.divider()
            
            # C. Top Result Banner (The "Verdict")
            top_match = results[0]
            if top_match['active']:
                st.markdown(f"""
                <div style="background-color: #004d2c; padding: 20px; border-radius: 5px; text-align: center; color: white; margin-bottom: 20px;">
                    <div style="font-size: 0.9rem; color: #c5a059; text-transform: uppercase; letter-spacing: 2px;">Primary Classification</div>
                    <div style="font-size: 2.5rem; font-family: serif; font-weight: bold;">{top_match['concern'].upper()}</div>
                    <div style="font-size: 1rem; opacity: 0.9;">Confidence: {top_match['score']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Analysis Inconclusive: No strong category match found.")

            # D. Two-Column Layout for details
            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("##### 🧬 Detected Actives")
                if found_ingredients:
                    for ing in found_ingredients:
                        st.markdown(f'<span class="ingredient-box">{ing}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No specific active ingredients identified.")

            with right_col:
                st.markdown("##### 📊 Full Profile")
                for res in results[:4]: # Show top 4 only for cleanliness
                    icon = "✅" if res['active'] else "⬜"
                    score_color = "#004d2c" if res['active'] else "#999"
                    
                    # Custom HTML Progress Bar for "Classic" look
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                            <span style="font-weight: bold; color: {score_color};">{res['concern'].title()}</span>
                            <span>{res['score']:.0%}</span>
                        </div>
                        <div style="background-color: #eee; height: 6px; border-radius: 3px; width: 100%;">
                            <div style="background-color: {score_color}; width: {res['score']*100}%; height: 6px; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.info("Please enter text to begin analysis.")
