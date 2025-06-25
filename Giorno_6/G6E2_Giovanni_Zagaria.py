from sentence_transformers import SentenceTransformer, util

# 1. Modello di embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Frasi da confrontare
sentences = [
    "Luca ha comprato una macchina nuova.",
    "Luca si è appena comprato una macchina nuova.",
    "Oggi piove molto a Milano."
]

# 3. Calcolo embedding
embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. Similarità coseno
similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1]).item()
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2]).item()

# 5. Output
print("Similarità tra frase 1 e 2:", similarity_1_2)
print("Similarità tra frase 1 e 3:", similarity_1_3)
