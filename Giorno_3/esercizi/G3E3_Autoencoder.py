
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

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

# Limita campione per velocità
MAX_CLIENTI = 1000
if len(agg) > MAX_CLIENTI:
    agg = agg.sample(MAX_CLIENTI, random_state=42).reset_index(drop=True)

# Standardizza le feature
features = ['total_spent', 'num_orders', 'num_categories', 'activity_days']
X = StandardScaler().fit_transform(agg[features])

# Costruisci autoencoder semplice
from tensorflow.keras import layers

input_dim = X.shape[1]
encoding_dim = 2  # Bottleneck a 2 dimensioni

with tqdm(total=1, desc="Costruzione Autoencoder") as pbar:
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='linear')(encoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    pbar.update(1)

# Addestramento (con barra progresso)
EPOCHS = 50
BATCH_SIZE = 32

with tqdm(total=EPOCHS, desc="Training Autoencoder") as pbar:
    history = autoencoder.fit(
        X, X,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: pbar.update(1))]
    )

# Codifica e ricostruzione
encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X)
X_decoded = autoencoder.predict(X)

# Visualizza bottleneck (2D)
plt.figure(figsize=(7,5))
sns.scatterplot(x=X_encoded[:,0], y=X_encoded[:,1], alpha=0.7)
plt.title("Rappresentazione compressa (bottleneck, 2D) dei clienti Olist")
plt.xlabel("Bottleneck dim 1")
plt.ylabel("Bottleneck dim 2")
plt.show()

# Visualizza errore di ricostruzione
reconstruction_error = ((X - X_decoded) ** 2).mean(axis=1)

plt.figure(figsize=(7,4))
plt.hist(reconstruction_error, bins=30, alpha=0.7, color='cornflowerblue')
plt.title("Errore di ricostruzione per cliente (autoencoder)")
plt.xlabel("Errore (MSE)")
plt.ylabel("Numero clienti")
plt.show()

print(f"Errore medio di ricostruzione: {reconstruction_error.mean():.3f}")

# Mostra esempio: dati originali e ricostruiti per i primi 5 clienti
df_show = pd.DataFrame(X[:5], columns=features)
df_recon = pd.DataFrame(X_decoded[:5], columns=[f"{f}_recon" for f in features])
df_compare = pd.concat([df_show, df_recon], axis=1)
print("Dati originali (standardizzati) e ricostruiti dai primi 5 clienti:")
print(df_compare.to_string(index=False))