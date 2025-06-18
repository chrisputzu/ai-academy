
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Scarica e carica dati Olist con tqdm
with tqdm(total=1, desc="Download Olist") as pbar:
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    pbar.update(1)

file_list = ['olist_orders_dataset.csv', 'olist_order_items_dataset.csv']
dfs = {}
for file in tqdm(file_list, desc="Lettura file"):
    dfs[file] = pd.read_csv(os.path.join(path, file))

orders = dfs['olist_orders_dataset.csv']
order_items = dfs['olist_order_items_dataset.csv']

# Calcola la spesa totale per cliente
order_items = order_items.merge(orders[['order_id', 'customer_id']], on='order_id')
agg = order_items.groupby('customer_id').agg({'price': 'sum'}).reset_index().rename(columns={'price': 'total_spent'})
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]  # outlier trimming

# Riduci a MAX_CLIENTI per velocità
MAX_CLIENTI = 4000
if len(agg) > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)
    print(f"Campione casuale ridotto a {MAX_CLIENTI} clienti per velocità.")

# Standardizza la feature
X = StandardScaler().fit_transform(agg[['total_spent']])

# Applica DBSCAN (eps e min_samples possono essere modificati in base al dataset)
print("Esecuzione clustering DBSCAN...")
db = DBSCAN(eps=0.5, min_samples=5)
agg['cluster'] = db.fit_predict(X)

# Conta i cluster trovati
n_cluster = len(set(agg['cluster'])) - (1 if -1 in agg['cluster'] else 0)
n_outlier = (agg['cluster'] == -1).sum()
print(f"Numero di cluster trovati da DBSCAN: {n_cluster}")
print(f"Numero di clienti 'outlier' (cluster -1): {n_outlier}")

# Visualizza risultato
plt.figure(figsize=(8,5))
sns.boxplot(data=agg, x='cluster', y='total_spent', palette='tab10')
plt.title('Distribuzione spesa totale per cluster DBSCAN')
plt.xlabel('Cluster (cluster -1 = outlier/noise)')
plt.ylabel('Spesa totale')
plt.show()