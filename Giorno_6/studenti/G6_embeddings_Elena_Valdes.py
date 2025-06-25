import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()
endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_API_KEY")

ai_client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview"
)

sentences = [
    'Il gatto dorme sul divano',
    'Il felino riposa sul divano',
    'Sta nevicando intensamente a Torino'
]

embeddings_list = [ai_client.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding for text in sentences]

for idx, vector in enumerate(embeddings_list, 1):
    print(f'\nEmbedding {idx}', vector)

# Riduzione PCA 3D per confronto direzionale
pca_model = PCA(n_components=3)
vectors_3d = pca_model.fit_transform(embeddings_list)

# Origine per i vettori
origin_point = np.zeros(3)

# Normalizzazione vettori (norma L2) per confronto solo direzioni
norm_values = np.linalg.norm(vectors_3d, axis=1, keepdims=True)
unit_vectors = vectors_3d / norm_values 

# Grafico 3D
figure = plt.figure(figsize=(8, 6))
axis = figure.add_subplot(111, projection='3d')

colors_palette = ['purple', 'orange', 'cyan']
labels_list = ['S1: Il gatto dorme sul divano',
               'S2: Il felino riposa sul divano',
               'S3: Sta nevicando intensamente a Torino']

# Vettori unitari dall'origine
for vec, col, lab in zip(unit_vectors, colors_palette, labels_list):
    axis.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                color=col, length=1, arrow_length_ratio=0.1, label=lab)
    axis.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, lab.split(":")[0], color=col)

#grafico
axis.set_title("Confronto Embeddings Frasi")
axis.set_xlim([-1, 1])
axis.set_ylim([-1, 1])
axis.set_zlim([-1, 1])
axis.set_xlabel('PC1')
axis.set_ylabel('PC2')
axis.set_zlabel('PC3')
axis.legend(loc='upper right')
plt.tight_layout()
plt.show()
