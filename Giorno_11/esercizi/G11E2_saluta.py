import os
from dotenv import load_dotenv

# Carica variabili da .env nella stessa directory
load_dotenv()

def saluta():
    api_key = os.environ.get("API_KEY")
    if api_key:
        print(f"Ciao! La tua chiave è: {api_key}")
    else:
        print("Ciao! Nessuna chiave fornita.")

if __name__ == "__main__":
    saluta()
