# 💊 MedicL Bot — Your AI Health Assistant

MedicL Bot is an intelligent AI-powered chatbot that takes symptom input (in multiple languages!) and suggests relevant medicines, dosages, home remedies, and warns about potential side effects or high-risk drugs.

Built with:
- 🧠 Machine Learning (Naive Bayes + Transformers)
- 🌐 Streamlit Web App UI
- 🈳 Multilingual support using Google Translate
- 📋 Feedback logging
- ✅ Safe-for-children and pregnancy checks


## 🛠️ How to Run It Locally

```bash
git clone https://github.com/kakarotcode/medicl-bot.git
cd medicl-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run medicl_streamlit.py