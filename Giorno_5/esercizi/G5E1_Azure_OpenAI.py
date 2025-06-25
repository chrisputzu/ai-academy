import openai

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="", # <-- Il tuo  API endpoint di Azure OpenAI
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