import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
from googletrans import Translator

if not os.path.exists("chat_logs.csv"):
    with open("chat_logs.csv", "w", encoding="utf-8") as f:
        f.write("user_input,translated_input ➝ matched_symptom,predicted_medicine,feedback\n")

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
translator = Translator()
st.title("💊 MedicL Bot")
st.write("Enter your symptoms or select from the list to get medicine recommendations.")

# Input section
user_input = st.text_input("📝 Type your symptoms:")
selected_symptom = st.selectbox("⬇️ Or select a symptom (optional):", [""] + symptom_list)

# Use selected_symptom if provided
query = user_input if user_input else selected_symptom

if query:
    translated = translator.translate(query, dest='en').text
    query_embed = st_model.encode(translated, convert_to_tensor=True)
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

    # Build metadata dictionary
    medicine_metadata = {}
    for _, row in df.iterrows():
        medicine_metadata[row['medicine_name']] = {
            "symptom_keywords": row['symptom_keywords'],
            "dosage": row['dosage'],
            "home_remedy": row['home_remedy'],
            "side_effects": row['side_effects'],
            "safe_for_children": row['medicine_name'] in ["Paracetamol 500mg", "Cetirizine 10mg", "Dolo 650"],
            "safe_during_pregnancy": row['medicine_name'] in ["Paracetamol 500mg", "Domstal 10mg"],
            "high_risk": row['medicine_name'] in ["Avastin 400mg Injection", "Norflox TZ + ORS"]
        }

    info = medicine_metadata[predicted_medicine]

    st.subheader("🤖 MedicBot Suggestion")
    st.markdown(f"**🩺 Medicine:** {predicted_medicine}")
    st.markdown(f"**💊 Dosage:** {info['dosage']}")
    st.markdown(f"**🪴 Home Remedy:** {info['home_remedy'] if pd.notna(info['home_remedy']) else 'Not available.'}")
    st.markdown(f"**⚠️ Side Effects:** {info['side_effects'] if pd.notna(info['side_effects']) else 'Not available.'}")

    if info['high_risk']:
        st.warning("⚠️ This medicine may be considered high-risk. Please consult a doctor before use.")
    if not info['safe_for_children']:
        st.info("👶 Not confirmed safe for children.")
    if not info['safe_during_pregnancy']:
        st.info("🤰 Not confirmed safe during pregnancy.")

    # Feedback section
    feedback = st.radio("📝 Was this suggestion helpful?", ("Yes", "No"), horizontal=True)
    submitted = st.button("Submit Feedback")
    if submitted:
        with open("chat_logs.csv", "a", encoding="utf-8") as f:
            f.write(f'"{user_input}","{translated} ➝ {selected}","{predicted_medicine}","{feedback}"\n')
        st.success("✅ Feedback submitted! Thank you.")