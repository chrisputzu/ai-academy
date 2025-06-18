
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# Scarica e carica dati Olist con tqdm
with tqdm(total=1, desc="Download Olist") as pbar:
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    pbar.update(1)

file_list = ['olist_orders_dataset.csv', 'olist_order_items_dataset.csv', 'olist_products_dataset.csv']
dfs = {}
for file in tqdm(file_list, desc="Lettura file"):
    dfs[file] = pd.read_csv(os.path.join(path, file))

orders = dfs['olist_orders_dataset.csv']
order_items = dfs['olist_order_items_dataset.csv']
products = dfs['olist_products_dataset.csv']

# Aggrega feature per cliente (modificabile)
order_items = order_items.merge(orders[['order_id', 'customer_id']], on='order_id')
order_items = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
agg = order_items.groupby('customer_id').agg(
    total_spent = ('price', 'sum'),
    num_orders = ('order_id', 'nunique'),
    num_categories = ('product_category_name', 'nunique')
).reset_index()

agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]  # Rimuovi outlier grossolani

# Riduci a MAX_CLIENTI per velocità e visualizzazione
MAX_CLIENTI = 200
if len(agg) > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)
    print(f"Campione casuale ridotto a {MAX_CLIENTI} clienti per velocità e chiarezza.")

# Prepara le feature e standardizza
features = ['total_spent', 'num_orders', 'num_categories']
X = StandardScaler().fit_transform(agg[features])

# Clustering gerarchico
n_clusters = 3  # Puoi cambiare qui o decidere dal dendrogramma
clustering = AgglomerativeClustering(n_clusters=n_clusters)
labels = clustering.fit_predict(X)
agg['cluster'] = labels

# Visualizza la distribuzione di una feature (ad es. total_spent) nei cluster
plt.figure(figsize=(8,5))
sns.violinplot(data=agg, x='cluster', y='total_spent', palette='tab10')
plt.title(f'Distribuzione spesa totale per cluster gerarchico (n={n_clusters})')
plt.xlabel('Cluster')
plt.ylabel('Spesa totale')
plt.show()

# Visualizza il dendrogramma per esplorazione visiva
plt.figure(figsize=(14, 6))
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Dendrogramma (ultimi 30 cluster)')
plt.xlabel('Cluster/Cliente')
plt.ylabel('Distanza')
plt.show()