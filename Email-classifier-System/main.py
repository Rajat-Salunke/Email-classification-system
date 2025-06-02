# main.py
from fastapi import FastAPI, Request
import spacy
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize FastAPI app
app = FastAPI()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset and train classifier (for demonstration)
# ⚠️ In real apps, load pre-trained model/vectorizer from files (e.g., joblib.load)
emails_df = pd.read_csv('emails.csv')
X = emails_df['email']
y = emails_df['type']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_vec, y)

# PII masking function
def mask_pii(text):
    doc = nlp(text)
    masked_email = text
    list_of_masked_entities = []
    
    regex_patterns = {
        'phone_number': r'\b\d{10}\b|\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b',
        'email': r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        'dob': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        'credit_debit_no': r'\b\d{16}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b\d{2}/\d{2}\b',
        'aadhar_num': r'\b\d{12}\b'
    }
    
    for entity, pattern in regex_patterns.items():
        for match in re.finditer(pattern, masked_email):
            start, end = match.span()
            entity_text = match.group()
            masked_email = masked_email.replace(entity_text, f"[{entity}]")
            list_of_masked_entities.append({
                "position": [start, end],
                "classification": entity,
                "entity": entity_text
            })
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entity_text = ent.text
            masked_email = masked_email.replace(entity_text, "[full_name]")
            list_of_masked_entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": entity_text
            })
    
    return masked_email, list_of_masked_entities

# Classification pipeline
def process_email(input_email_body):
    masked_email, list_of_masked_entities = mask_pii(input_email_body)
    email_vec = vectorizer.transform([masked_email])
    predicted_category = classifier.predict(email_vec)[0]
    return {
        "input_email_body": input_email_body,
        "list_of_masked_entities": list_of_masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_category
    }
app = FastAPI()
@app.post("/classify")
async def classify_email(request: Request):
    data = await request.json()
    input_email_body = data.get("input_email_body", "")
    output = process_email(input_email_body)
    return output

