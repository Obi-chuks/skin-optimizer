
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
st.set_page_config(page_title="L'Oréal Skin AI", layout="centered")

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
    st.error("⚠️ deployment_pack.pkl not found!")
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

# --- UI ---
st.title("🧪 L'Oréal Skincare Optimizer")
st.caption("AI-Powered Multi-Label Classification System")

input_text = st.text_area("Product Description:", height=150, 
                         placeholder="e.g. La Roche-Posay Effaclar Serum with Salicylic Acid...")

if st.button("Analyze Product", type="primary"):
    if input_text:
        with st.spinner("Analyzing chemical composition..."):
            X_input = process_input(input_text)
            probs = model.predict_proba(X_input)[0]

            st.subheader("Results")
            cols = st.columns(len(TARGET_COLS))
            for i, col in enumerate(TARGET_COLS):
                score = probs[i]
                is_active = score > best_thresh[i]
                with cols[i]:
                    if is_active:
                        st.metric(col.upper(), f"{score:.0%}", delta="Detected")
                    else:
                        st.metric(col.upper(), f"{score:.0%}", delta_color="off")
    else:
        st.warning("Please enter text first.")
