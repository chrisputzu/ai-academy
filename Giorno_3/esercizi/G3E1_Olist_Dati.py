import kagglehub
import pandas as pd
import os
from tqdm import tqdm

# Scarica il dataset Olist
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
print(f"Dati scaricati in: {path}")

# Elenca tutti i file CSV nel dataset
files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"File trovati:\n{files}")

# Crea cartella di output locale se non esiste
out_dir = "olist_completo"
os.makedirs(out_dir, exist_ok=True)

# Carica e salva tutti i file come CSV (puoi cambiarlo in parquet se vuoi)
for file in tqdm(files, desc="Salvataggio CSV"):
    df = pd.read_csv(os.path.join(path, file))
    out_path = os.path.join(out_dir, file)
    df.to_csv(out_path, index=False)
    print(f"Salvato: {out_path}")

print("\nTutti i file Olist sono stati salvati nella cartella 'olist_completo'")