from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline    
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAI
import openai


from dotenv import load_dotenv
import os

# Carica le variabili di ambiente dal file .env
load_dotenv()
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')


# 1. Carica documenti

files = ["Giorno_5/Mail.txt", "Giorno_5/nota_di_credito.txt", "Giorno_5/ordine_di_acquisto.txt", "Giorno_5/Fattura.txt"]
documents = []
for file in files:
    loader = TextLoader(file, encoding="utf-8")
    documents.extend(loader.load())

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(chunks, embedding_model)

# 4. Retrieval
retriever = vectordb.as_retriever()


# 5. Configurazione del modello Azure OpenAI 
# Configura il client Azure OpenAI
llm = AzureOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    model="gpt-demo",  
    api_version="2025-01-01-preview", 
)


# 6. Retrieval + QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Domanda
query = "Qual è il nuovo IBAN comunicato da Mario Rossi?"
risposta = qa_chain.invoke({"query": query})

print(risposta)

# 8. Output
print("Risultato:", risposta["result"])
print("\nFonte:")
for doc in risposta["source_documents"]:
    print("-", doc.metadata["source"])
