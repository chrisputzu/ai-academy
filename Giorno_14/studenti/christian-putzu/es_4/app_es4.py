import kagglehub
import pandas as pd
import os
import nltk
import shutil
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Download NLTK resources if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# 2. Prepare the dataset directory and download the dataset if needed
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created directory: {dataset_dir}")

local_csv_path = os.path.join(dataset_dir, "augmented_spam.csv")
if os.path.exists(local_csv_path):
    print(f"Dataset already exists at: {local_csv_path}")
    csv_path = local_csv_path
else:
    print("Downloading SMS Spam Collection dataset...")
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    print(f"Dataset downloaded to: {path}")
    original_csv_path = os.path.join(path, "augmented_spam.csv")
    shutil.copy2(original_csv_path, local_csv_path)
    print(f"Dataset copied to: {local_csv_path}")
    csv_path = local_csv_path

# 3. Load and clean the data
df = pd.read_csv(csv_path, encoding='latin-1')
df_cleaned = df[['v1', 'v2']].copy()
df_cleaned['v2'] = df_cleaned['v2'].astype(str).apply(lemmatize_text)

# 4. Split the dataset into train and test sets
train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(train_data['v2'])
test_features = vectorizer.transform(test_data['v2'])

# 5. Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_features, train_data['v1'])

# 6. Evaluate the model on the test set
predictions = model.predict(test_features)
accuracy = accuracy_score(test_data['v1'], predictions)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(test_data['v1'], predictions))

# 7. LIME explanation for 5 random test records
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt

# Define class names for LIME
class_names = list(df_cleaned['v1'].unique())  # e.g., ['ham', 'spam']

# Function for LIME to get predicted probabilities
def predict_proba(texts):
    features = vectorizer.transform(texts)
    return model.predict_proba(features)

explainer = LimeTextExplainer(class_names=class_names)

num_records = 5
selected_indices = np.random.choice(test_data.index, size=num_records, replace=False)
selected_records = test_data.loc[selected_indices]

summary_table = []
for idx, row in selected_records.iterrows():
    text = row['v2']
    real_label = row['v1']
    pred_label = model.predict(vectorizer.transform([text]))[0]
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6,
        labels=[class_names.index(pred_label)]
    )
    print(f"\nRecord: {text}")
    print(f"True Label: {real_label} | Predicted: {pred_label}")
    print("Top features (word, contribution):")
    top_words = []
    for word, weight in exp.as_list(label=class_names.index(pred_label)):
        top_words.append(f'"{word}" ({weight:+.2f})')
        print(f'  {word}: {weight:+.2f}')
    summary_table.append({
        "Text": text,
        "True Label": real_label,
        "Predicted": pred_label,
        "Top Features": ", ".join(top_words)
    })
    # Plot local LIME explanation for each record
    fig = exp.as_pyplot_figure(label=class_names.index(pred_label))
    plt.title(f"LIME explanation for: {pred_label}")
    plt.tight_layout()
    plt.show()

# Create and display a summary table
summary_df = pd.DataFrame(summary_table)
print("\n=== Summary Table ===")
print(summary_df[["Text", "Predicted", "Top Features"]])

# 8. (Optional) Test model on custom messages
try:
    with open('../../../esercizi/messaggi test.txt', 'r') as file:
        test_message = file.read()
    splitted_list = test_message.split('\n')
    for i in splitted_list:
        lemmatized = lemmatize_text(i)
        test_features = vectorizer.transform([lemmatized])
        predicted_label = model.predict(test_features)
        print(f"\nTest Message: {i}")
        print(f"Predicted Label: {predicted_label[0]}")
except Exception as e:
    print(f"Custom test file not found or error: {e}")

# Record: URGENT! You win money! Call now! Message 332
# True Label: spam | Predicted: spam
# Top features (word, contribution):
#   win: -0.33
#   URGENT: -0.31
#   Message: -0.08
#   money: -0.02
#   Call: -0.01
#   now: -0.00

# Le parole "win" e "URGENT" sono forti indicatori di spam, con elevati contributi negativi
# Altre parole come "money" e "Call" contribuiscono alla classificazione spam ma con minor impatto