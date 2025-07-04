
import kagglehub
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Scarica e carica i dati Olist (con progress bar)
with tqdm(total=1, desc="Download Olist") as pbar:
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    pbar.update(1)
file_list = ['olist_orders_dataset.csv', 'olist_order_items_dataset.csv']
dfs = {}
for file in tqdm(file_list, desc="Lettura file"):
    dfs[file] = pd.read_csv(os.path.join(path, file))
orders = dfs['olist_orders_dataset.csv']
order_items = dfs['olist_order_items_dataset.csv']

# Prepara la feature: spesa totale per cliente
order_items = order_items.merge(orders[['order_id', 'customer_id']], on='order_id')
agg = order_items.groupby('customer_id').agg({'price': 'sum'}).reset_index().rename(columns={'price': 'total_spent'})
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]

# Stampa il numero totale di clienti
n_clienti = len(agg)
print(f"\nNumero totale di clienti nel database Olist: {n_clienti}")

# Taglia il dataset per motivi di velocità (ad es. 2000 clienti)
MAX_CLIENTI = 50000
if n_clienti > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)
    print(f"Campione casuale ridotto a {MAX_CLIENTI} clienti per velocità.")

# Continua con il clustering come prima
X = StandardScaler().fit_transform(agg[['total_spent']])

ks = range(2, 7)
sil_scores = []
print("Clustering e valutazione silhouette...")
for k in tqdm(ks, desc="Clusterizzazione"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

plt.figure(figsize=(7,4))
plt.plot(list(ks), sil_scores, marker='o')
plt.title('Silhouette Score per diversi k')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Silhouette Score medio')
plt.grid()
plt.show()

best_k = list(ks)[int(np.argmax(sil_scores))]
print(f'Miglior numero di cluster secondo silhouette score: {best_k}')

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
agg['cluster'] = km.fit_predict(X)
score = silhouette_score(X, agg['cluster'])

plt.figure(figsize=(8,5))
sns.violinplot(data=agg, x='cluster', y='total_spent', palette='tab10')
plt.title(f'Spesa totale per cluster (k={best_k}), silhouette score medio={score:.2f}')
plt.xlabel('Cluster')
plt.ylabel('Spesa totale')
plt.show()