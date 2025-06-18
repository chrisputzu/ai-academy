
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Scarica e carica dati Olist
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

# Crea feature per cliente
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
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]  # outlier trimming

# Limita il campione per velocità
MAX_CLIENTI = 2000
if len(agg) > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)

# Standardizza le feature
features = ['total_spent', 'num_orders', 'num_categories', 'activity_days']
X = StandardScaler().fit_transform(agg[features])

# Applica PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Stampa quanta varianza spiega ogni componente
import matplotlib.pyplot as plt

print("Varianza spiegata da ogni componente:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.2%}")

plt.figure(figsize=(7,4))
plt.bar(range(1, 5), pca.explained_variance_ratio_, tick_label=[f'PC{i}' for i in range(1,5)])
plt.ylabel('Varianza spiegata')
plt.xlabel('Componente principale')
plt.title('Varianza spiegata da ciascuna componente PCA')
plt.show()

# Puoi ridurre le dimensioni mantenendo solo le prime 2 (spesso spiegano >80% della varianza)
X_pca2 = X_pca[:, :2]