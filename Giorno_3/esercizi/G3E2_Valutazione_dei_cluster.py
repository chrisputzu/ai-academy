
import kagglehub
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Scarica e carica i dati Olist
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
order_items = pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv'))

# Prepara la feature: spesa totale per cliente
order_items = order_items.merge(orders[['order_id', 'customer_id']], on='order_id')
agg = order_items.groupby('customer_id').agg({'price': 'sum'}).reset_index().rename(columns={'price': 'total_spent'})
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]

X = StandardScaler().fit_transform(agg[['total_spent']])

# Progress bar con tqdm standard
sil_scores = []
ks = range(2, 8)
for k in tqdm(ks, desc="Clustering e valutazione silhouette"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

# Visualizza la curva silhouette score
plt.figure(figsize=(7,4))
plt.plot(ks, sil_scores, marker='o')
plt.title('Silhouette Score per diversi k')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Silhouette Score medio')
plt.grid()
plt.show()

# Trova il miglior k
best_k = ks[np.argmax(sil_scores)]
print(f'Miglior numero di cluster secondo silhouette score: {best_k}')

# Applica k-Means con k ottimale
km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=100)
agg['cluster'] = km.fit_predict(X)

# Visualizza distribuzione cluster (violin plot)
plt.figure(figsize=(8,5))
sns.violinplot(data=agg, x='cluster', y='total_spent', palette='tab10')
plt.title(f'Distribuzione spesa totale nei cluster (k={best_k})')
plt.show()

# Mostra silhouette score nei cluster (opzionale, rapido)
agg['silhouette'] = silhouette_samples(X, agg['cluster'])
plt.figure(figsize=(8,4))
sns.boxplot(data=agg, x='cluster', y='silhouette', palette='tab10')
plt.title('Distribuzione silhouette score nei cluster')
plt.show()