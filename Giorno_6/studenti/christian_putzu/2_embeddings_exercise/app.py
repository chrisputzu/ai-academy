import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_api_key = os.getenv("AZURE_API_KEY")

client = AzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview"
)

phrases = [
    'Luca ha comprato una macchina nuova',
    'Luca si è comprato una macchina nuova',
    'Oggi piove molto a Milano'
]

# Embeddings using text-embedding-ada-002
embeds = [client.embeddings.create(input=p, model="text-embedding-ada-002").data[0].embedding for p in phrases]

for i, emb in enumerate(embeds, 1):
    print(f'\nEmbedding {i}', emb)

# Dimensionality reduction to 3D for vector direction comparison
pca = PCA(n_components=3)
e3d = pca.fit_transform(embeds)

# Origin for all vectors
origin = np.zeros(3)

# Normalize vectors (L2 norm) to compare directions only
norms = np.linalg.norm(e3d, axis=1, keepdims=True)
u = e3d / norms  # unit vectors

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'green', 'red']
labels = ['F1: Luca ha comprato una macchina nuova',
          'F2: Luca si è comprato una macchina nuova',
          'F3: Oggi piove molto a Milano']

# Unit vectors from origin
for vec, c, lab in zip(u, colors, labels):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              color=c, length=1, arrow_length_ratio=0.1, label=lab)
    ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, lab.split(":")[0], color=c)

# Chart settings
ax.set_title("Direction Comparison of Phrase Embeddings")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
