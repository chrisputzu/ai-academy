
import kagglehub
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Scarica e carica i dati
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
order_items = pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(path, 'olist_products_dataset.csv'))

# Prepara feature cliente base: total_spent, num_orders, num_categories
order_items = order_items.merge(orders[['order_id', 'customer_id']], on='order_id')
order_items = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
agg = order_items.groupby('customer_id').agg({
    'order_id': 'nunique',
    'price': 'sum',
    'product_category_name': 'nunique'
}).reset_index().rename(columns={
    'order_id': 'num_orders',
    'price': 'total_spent',
    'product_category_name': 'num_categories'
})
agg = agg[agg['total_spent'] < agg['total_spent'].quantile(0.99)]
agg['avg_ticket'] = agg['total_spent'] / agg['num_orders']

# Clustering k-Means su clienti
features = ['num_orders', 'total_spent', 'num_categories', 'avg_ticket']
X = StandardScaler().fit_transform(agg[features])
kmeans = KMeans(n_clusters=4, random_state=42)
agg['cluster'] = kmeans.fit_predict(X)

# 1. Pino: total_spent (sbilanciato)
plt.figure(figsize=(7,4))
sns.violinplot(data=agg, x='cluster', y='total_spent', palette='tab10')
plt.title('1. Forma a pino (spesa totale, distribuzione sbilanciata)')
plt.show()

# 2. Fagiolo: log(total_spent) (più simmetrica, quasi gaussiana)
agg['log_spent'] = np.log1p(agg['total_spent'])
plt.figure(figsize=(7,4))
sns.violinplot(data=agg, x='cluster', y='log_spent', palette='tab10')
plt.title('2. Forma a fagiolo (log spesa totale, più simmetrica)')
plt.show()

# 3. Pettine: num_categories (discreta, pochi valori possibili)
plt.figure(figsize=(7,4))
sns.violinplot(data=agg, x='cluster', y='num_categories', palette='tab10')
plt.title('3. Forma a pettine (numero categorie, variabile discreta)')
plt.show()

# 4. Fagiolo largo su filtrato: total_spent clienti attivi (almeno 5 ordini)
filtered = agg[agg['num_orders'] >= 5]
plt.figure(figsize=(7,4))
sns.violinplot(data=filtered, x='cluster', y='total_spent', palette='tab10')
plt.title('4. Fagiolo largo (clienti attivi: almeno 5 ordini)')
plt.show()