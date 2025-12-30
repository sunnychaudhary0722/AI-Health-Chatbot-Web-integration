# 🏥 AI Health Chatbot – Web Integration

## 📌 Project Overview
The **AI Health Chatbot** is an end-to-end **Machine Learning and NLP-based web application** that predicts possible diseases based on user-provided symptoms. It processes natural language input, intelligently handles typos and synonyms, asks follow-up questions, and provides disease predictions along with confidence scores, descriptions, and precautionary health advice.  

This project demonstrates the practical use of **AI in preliminary healthcare guidance** and is intended for educational purposes only, not as a replacement for professional medical consultation.

### ✨ Key Features
- Symptom extraction from natural language input  
- Intelligent typo correction and synonym matching using NLP  
- Disease prediction using a **Random Forest Classifier**  
- Interactive chatbot with dynamic follow-up questions  
- Prediction confidence score  
- Disease description and precautionary recommendations  
- Motivational health quote at the end of interaction  

### 🛠️ Technologies Used
Python, Pandas, NumPy, Scikit-learn, Natural Language Processing (NLP), Random Forest Classifier, Flask, HTML, CSS, JavaScript, CSV-based medical datasets.

### ⚙️ Working Flow
User symptoms are processed using NLP techniques to extract relevant keywords. The chatbot asks follow-up questions to improve accuracy, after which a trained Random Forest model predicts the most probable disease. The final output includes a confidence score, disease description, and precautionary advice.

### 🚀 Run Locally
```bash
git clone https://github.com/sunnychaudhary0722/AI-Health-Chatbot-Web-integration.git
cd AI-Health-Chatbot-Web-integration
pip install -r requirements.txt
python app.py
