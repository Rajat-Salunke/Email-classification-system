import pandas as pd

# Load CSV file
df = pd.read_csv('emails.csv')  # Change filename as per your download

# Display first few rows
print(df.head())

# Show column names
print(df.columns)


import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to mask PII in an email text
def mask_pii(text):
    doc = nlp(text)
    masked_email = text
    list_of_masked_entities = []

    # Regex-based patterns for sensitive data
    regex_patterns = {
        'phone_number': r'\b\d{10}\b|\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b',
        'email': r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        'dob': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        'credit_debit_no': r'\b\d{16}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b\d{2}/\d{2}\b',
        'aadhar_num': r'\b\d{12}\b'
    }

    # Apply regex masking
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

    # spaCy-based masking for names (using PERSON entity)
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
