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

# Download required NLTK resources (only the first time)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Create dataset directory if it doesn't exist
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created directory: {dataset_dir}")

# Define the final path for our dataset
local_csv_path = os.path.join(dataset_dir, "augmented_spam.csv")

# Check if dataset already exists locally
if os.path.exists(local_csv_path):
    print(f"Dataset already exists at: {local_csv_path}")
    csv_path = local_csv_path
else:
    # Download the dataset
    print("Downloading SMS Spam Collection dataset...")
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Copy the dataset to our local dataset directory
    original_csv_path = os.path.join(path, "augmented_spam.csv")
    shutil.copy2(original_csv_path, local_csv_path)
    print(f"Dataset copied to: {local_csv_path}")
    csv_path = local_csv_path

# Load the dataset into a DataFrame
df = pd.read_csv(csv_path, encoding='latin-1')
print(df)

# Display basic information about the DataFrame
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Check for any missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

df_cleaned = df[['v1', 'v2']]
print(df_cleaned)

# Apply lemmatization to all messages
df_cleaned['v2'] = df_cleaned['v2'].astype(str).apply(lemmatize_text)

train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')

train_features = vectorizer.fit_transform(train_data['v2'])
test_features = vectorizer.transform(test_data['v2'])

print(f"\nTrain features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_features, train_data['v1'])

# Make predictions
predictions = model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_data['v1'], predictions)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(test_data['v1'], predictions))

# Test the model on custom messages
with open('../../../esercizi/messaggi test.txt', 'r') as file:
    test_message = file.read()
splitted_list = test_message.split('\n')

for i in splitted_list:
    # Apply lemmatization to each test message
    lemmatized = lemmatize_text(i)
    test_features = vectorizer.transform([lemmatized])
    predicted_label = model.predict(test_features)

    print(f"\nTest Message: {i}")
    print(f"Predicted Label: {predicted_label[0]}")



