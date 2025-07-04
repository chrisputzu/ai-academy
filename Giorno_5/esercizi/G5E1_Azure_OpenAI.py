import openai
from dotenv import load_dotenv
import os

# Carica le variabili di ambiente dal file .env
load_dotenv()
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')


# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key=azure_openai_api_key,  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint=azure_openai_endpoint, # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2025-01-01-preview",  # <-- La versione dal portale Azure
)


response = client.chat.completions.create(
    model="chatgpt-demo",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": "Qual è la capitale dell'Italia?"}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)