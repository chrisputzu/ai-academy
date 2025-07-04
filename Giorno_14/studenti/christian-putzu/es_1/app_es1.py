import kagglehub
import pandas as pd
import os
import nltk
import shutil
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

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

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Dictionary to store results
results = {}

print(f"\n{'='*60}")
print("TRAINING AND EVALUATING MODELS")
print(f"{'='*60}")

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(train_features, train_data['v1'])
    
    # Make predictions
    predictions = model.predict(test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(test_data['v1'], predictions)
    precision = precision_score(test_data['v1'], predictions, pos_label='spam')
    recall = recall_score(test_data['v1'], predictions, pos_label='spam')
    f1 = f1_score(test_data['v1'], predictions, pos_label='spam')
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(test_data['v1'], predictions))
    print("-" * 50)

# Create comparison visualization
print(f"\n{'='*60}")
print("CREATING COMPARISON VISUALIZATION")
print(f"{'='*60}")

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T
print("\nComparison Summary:")
print(results_df.round(4))

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Create single bar plot for comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('Model Performance Comparison - Spam Detection', fontsize=18, fontweight='bold', y=0.95)

# All metrics comparison with grouped bars
x = range(len(results_df.index))
width = 0.18  # Slightly narrower bars for better spacing
metrics = results_df.columns
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, metric in enumerate(metrics):
    positions = [pos + width * (i - 1.5) for pos in x]
    ax.bar(positions, results_df[metric], width, 
           label=metric, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=0, ha='center', fontsize=11)
ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)

# Add value labels on bars with better positioning
for i, metric in enumerate(metrics):
    for j, model in enumerate(results_df.index):
        value = results_df.loc[model, metric]
        x_pos = j + width * (i - 1.5)
        ax.text(x_pos, value + 0.02, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))

# Adjust layout to prevent overlapping
plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12)
plt.show()

# Print best performing model
best_model = results_df['F1-Score'].idxmax()
best_f1 = results_df.loc[best_model, 'F1-Score']

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"🏆 Best performing model: {best_model}")
print(f"🎯 Best F1-Score: {best_f1:.4f}")
print(f"\nFull performance of {best_model}:")
for metric, score in results_df.loc[best_model].items():
    print(f"  {metric}: {score:.4f}")
print(f"{'='*60}")

# # Test the model on custom messages
# with open('../../../esercizi/messaggi test.txt', 'r') as file:
#     test_message = file.read()
# splitted_list = test_message.split('\n')

# for i in splitted_list:
#     # Apply lemmatization to each test message
#     lemmatized = lemmatize_text(i)
#     test_features = vectorizer.transform([lemmatized])
#     predicted_label = model.predict(test_features)

#     print(f"\nTest Message: {i}")
#     print(f"Predicted Label: {predicted_label[0]}")



