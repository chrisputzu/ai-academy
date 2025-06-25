import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
azure_endpoint = os.getenv("AZURE_ENDPOINT_4o")
azure_api_key  = os.getenv("AZURE_API_KEY_4o")

client = AzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview",
)

DOCUMENTS_DIR = r"Giorno_5\RAGDocs" 
ALLOWED_TYPES = ["credit note", "purchase order", "contract", "email", "unknown"]

for filename in os.listdir(DOCUMENTS_DIR):
    path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.isfile(path):
        continue

    with open(path, "r", encoding="utf-8") as fp:
        doc_text = fp.read()

    prompt = f"""
    You will receive the full text of a company document.
    You have to return only the type of document, exactly as one of the options:
    {', '.join(ALLOWED_TYPES)}.
    \"\"\"{doc_text}\"\"\"    
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a document classification assistant specialised in classifying company documents."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=10,
        temperature=0
    )

    print(f"Input file: {filename}", "->", "Classified type: ", response.choices[0].message.content.strip())
