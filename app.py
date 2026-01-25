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

# --- CONFIG ---
st.set_page_config(page_title="L'Oréal Skin AI", page_icon="🧪", layout="centered")

# --- CUSTOM CSS FOR "PREMIUM" LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .highlight-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DEFINE THE TOKENIZER FUNCTION (CRITICAL) ---
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

# --- PREPROCESSING ---
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
    # Simple keyword search for display purposes
    ingredients = {
        "Salicylic Acid": ["salicylic", "bha"],
        "Retinol": ["retinol", "retinoid"],
        "Vitamin C": ["vitamin c", "ascorbic"],
        "Hyaluronic Acid": ["hyaluronic", "sodium hyaluronate"],
        "Niacinamide": ["niacinamide"],
        "Glycerin": ["glycerin"],
        "SPF": ["spf", "sunscreen"],
        "Clay": ["clay", "kaolin", "charcoal"]
    }
    found = []
    text_lower = text.lower()
    for name, keywords in ingredients.items():
        if any(k in text_lower for k in keywords):
            found.append(name)
    return found

# --- UI LAYOUT ---
st.title("🧪 L'Oréal Skincare Optimizer")
st.markdown("### Intelligent Product Composition Analysis")
st.caption("Paste any product description below to detect its target skin concerns and key ingredients.")

input_text = st.text_area("Product Description:", height=150, 
                         placeholder="Paste text here (e.g., 'La Roche-Posay Effaclar Medicated Gel Cleanser...')")

if st.button("Analyze Product DNA", type="primary"):
    if input_text:
        with st.spinner("Decoding formula..."):
            # 1. Run Model
            X_input = process_input(input_text)
            probs = model.predict_proba(X_input)[0]
            
            # 2. Get Ingredients
            found_ingredients = find_ingredients(input_text)

            # 3. Sort Results
            results = []
            for i, col in enumerate(TARGET_COLS):
                results.append({"concern": col, "score": probs[i], "active": probs[i] > best_thresh[i]})
            
            # Sort by score descending
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            top_match = results[0]

            # --- RESULT DISPLAY ---
            
            # A. The "Headline" Result
            st.divider()
            if top_match["score"] > 0.4:
                st.success(f"✅ **Primary Match: {top_match['concern'].upper()}** ({top_match['score']:.1%})")
            else:
                st.warning("⚠️ **Low Confidence:** This product description is vague.")

            # B. Ingredient Spotlight
            if found_ingredients:
                st.markdown
