
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

# Prepara le feature per cliente
order_items = order_items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id')
order_items = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
agg = order_items.groupby('customer_id').agg(
    total_spent=('price', 'sum'),
    num_orders=('order_id', 'nunique'),
    num_categories=('product_category_name', 'nunique'),
    first_order=('order_purchase_timestamp', 'min'),
    last_order=('order_purchase_timestamp', 'max')
).reset_index()
agg['activity_days'] = (pd.to_datetime(agg['last_order']) - pd.to_datetime(agg['first_order'])).dt.days + 1
agg = agg.drop(['first_order', 'last_order'], axis=1)
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]

# Campionamento per velocità
MAX_CLIENTI = 1500
if len(agg) > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)

# Standardizza le feature
features = ['total_spent', 'num_orders', 'num_categories', 'activity_days']
X = StandardScaler().fit_transform(agg[features])

# UMAP con tqdm
print("Applico UMAP...")
with tqdm(total=1, desc="UMAP") as pbar:
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=42)
    X_umap = reducer.fit_transform(X)
    pbar.update(1)

# Visualizzazione scatterplot UMAP
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], alpha=0.6)
plt.title('Visualizzazione UMAP dei clienti Olist')
plt.xlabel('UMAP dim 1')
plt.ylabel('UMAP dim 2')
plt.show()