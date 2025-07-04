import openai
import os

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="EXrbgf59diPUELQgaRYiStaEBg7D1Dr8JjrpfGPQ96PlVQtz4vxRJQQJ99BFACqBBLyXJ3w3AAAAACOGtys4",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="https://palmi-mc4maon6-southeastasia.openai.azure.com/", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2025-01-01-preview",  # <-- La versione dal portale Azure
)

paths = [
    'Giorno_5/Mail.txt',
    'Giorno_5/nota_di_credito.txt',
    'Giorno_5/ordine_di_acquisto.txt',
    'Giorno_5/Fattura.txt'
]


for path in paths:

    with open(path, 'r', encoding='utf-8') as file:
        testo = file.read()

    prompt = f"""
            Leggi il seguente testo e indica il tipo di documento tra: email, contratto, ordine di acquisto, nota di credito, fattura.

            Testo:
            \"\"\"{testo}\"\"\"

            Rispondi solo con il tipo di documento.
            """

    response = client.chat.completions.create(
        model="chatgpt-demo",
        messages=[
            {"role": "system", "content": "Sei un assistente AI."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=256,
        temperature=1,
    )

    predizione = response.choices[0].message.content.strip()
    print(f"Il file {path} è di tipo: {predizione}")
