"""
Configurazioni per il sistema di anonimizzazione documenti.
"""

import os
from dotenv import load_dotenv

# Carica variabili d'ambiente da .env se esiste
load_dotenv()

class Config:
    """Configurazione del sistema"""
    
    # Modelli AI
    NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
    
    # Azure OpenAI
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_ENDPOINT_EMB", os.getenv("AZURE_ENDPOINT"))
    AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_API_KEY_EMB", os.getenv("AZURE_API_KEY"))
    AZURE_API_VERSION = "2024-02-01"
    DEPLOYMENT_NAME = "gpt-4o"
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"

# Pattern regex per entità sensibili
REGEX_PATTERNS = {
    "IBAN": r'\bIT\d{2}(?: ?[A-Z0-9]){11,30}\b',
    "EMAIL": r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
    "CF": r'\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b',
    "CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "PHONE": r'\b\+?[0-9\s\-\(\)]{8,15}\b'
}

# Configura OPENAI_API_KEY per compatibilità
if Config.AZURE_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.AZURE_API_KEY