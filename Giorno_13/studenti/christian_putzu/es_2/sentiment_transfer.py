import pandas as pd
from transformers import pipeline

# Load the dataset
input_path = 'sentiment_dataset.csv'
df = pd.read_csv(input_path, header=None, names=['text', 'label', 'sentiment'])

# Foundation Model for sentiment analysis (only for ham)
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def get_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 characters
    label = result['label'].lower()
    if label == 'positive':
        return 'positivo'
    elif label == 'negative':
        return 'negativo'
    else:
        return 'neutro'

# New column for the new label
new_labels = []
for idx, row in df.iterrows():
    if row['label'].lower() == 'spam':
        new_labels.append('negativo')
    else:
        # Use the Foundation Model for ham messages
        sentiment = get_sentiment(row['text'])
        new_labels.append(sentiment)

df['new_sentiment'] = new_labels

# Save the new dataset
output_path = 'sentiment_dataset_labeled.csv'
df.to_csv(output_path, index=False)

# Transfer learning explanation
print("""
Transfer learning method used: Knowledge Distillation.
A large model (Foundation Model, e.g. DistilBERT) is used to label data or to instruct a smaller model (student model).
In this script, the Foundation Model labels ham messages with sentiment, while spam messages are directly labeled as negative.
""") 