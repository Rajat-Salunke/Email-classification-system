from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Dataset: email body & category
X = df['email']
y = df['type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_vec, y_train)

# Evaluate
y_pred = classifier.predict(X_test_vec)
print(classification_report(y_test, y_pred))

def process_email(input_email_body):
    # 1. Mask the email
    masked_email, list_of_masked_entities = mask_pii(input_email_body)

    # 2. Classify
    email_vec = vectorizer.transform([masked_email])
    predicted_category = classifier.predict(email_vec)[0]

    # 3. Format output
    output = {
        "input_email_body": input_email_body,
        "list_of_masked_entities": list_of_masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_category
    }
    return output

# Example
test_email = "Hello, my name is John Doe. My email is john@example.com and my phone is 123-456-7890. I have a billing issue."
result = process_email(test_email)
print(result)

