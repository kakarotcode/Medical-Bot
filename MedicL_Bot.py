import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load data
df = pd.read_csv("Bot_Dataset_CLEANED.csv")

# Model training
X = df['symptom_keywords']
y = df['medicine_name']
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Sentence transformer setup
symptom_list = df['symptom_keywords'].tolist()
st_model = SentenceTransformer('all-MiniLM-L6-v2')
symptom_embeddings = st_model.encode(symptom_list, convert_to_tensor=True)

# App UI
st.set_page_config(page_title="MedicL Bot", page_icon="💊")
st.title("💊 MedicL Bot")
st.write("Enter your symptoms or select from the list to get medicine recommendations.")

# Input section
user_input = st.text_input("📝 Type your symptoms:")
selected_symptom = st.selectbox("⬇️ Or select a symptom (optional):", [""] + symptom_list)

# Use selected_symptom if provided
query = user_input if user_input else selected_symptom

if query:
    query_embed = st_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embed, symptom_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=3)
    top_scores = top_results.values.tolist()
    top_indices = top_results.indices.tolist()

    st.subheader("🔍 Top Matches")
    options = []
    for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
        label = f"{symptom_list[idx]} ({score * 100:.2f}%)"
        options.append((label, symptom_list[idx]))
    match = st.radio("Select the most relevant match:", [opt[0] for opt in options])
    selected = next(symptom for label, symptom in options if label == match)

    # Prediction
    predicted_medicine = model.predict([selected])[0]
    row = df[df['medicine_name'] == predicted_medicine].iloc[0]

    # Risk checks
    high_risk_meds = ["Avastin 400mg Injection", "Norflox TZ + ORS"]
    safe_for_children = ["Paracetamol 500mg", "Cetirizine 10mg", "Dolo 650"]
    safe_during_pregnancy = ["Paracetamol 500mg", "Domstal 10mg"]

    st.subheader("🤖 MedicBot Suggestion")
    st.markdown(f"**🩺 Medicine:** {row['medicine_name']}")
    st.markdown(f"**💊 Dosage:** {row['dosage']}")
    st.markdown(f"**🪴 Home Remedy:** {row['home_remedy'] if pd.notna(row['home_remedy']) else 'Not available.'}")
    st.markdown(f"**⚠️ Side Effects:** {row['side_effects'] if pd.notna(row['side_effects']) else 'Not available.'}")

    if predicted_medicine in high_risk_meds:
        st.warning("⚠️ This medicine may be considered high-risk. Please consult a doctor before use.")
    if predicted_medicine not in safe_for_children:
        st.info("👶 Not confirmed safe for children.")
    if predicted_medicine not in safe_during_pregnancy:
        st.info("🤰 Not confirmed safe during pregnancy.")