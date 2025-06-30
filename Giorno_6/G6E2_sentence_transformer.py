from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Luca ha comprato una macchina nuova",
    "Luca si è appena comprato una macchina nuova",
    "Oggi piove molto a Milano"
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)

# 3. Calculate cosine similarities between the embeddings
similarities = model.similarity(embeddings, embeddings)

print("Similarità tra le frasi:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Frase {i+1} e Frase {j+1}: {similarities[i][j]:.4f}")