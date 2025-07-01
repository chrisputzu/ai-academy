# esercizo: progetta una pipeline rag con langchain

# permette di collegare in pochi passaggi:
# 1. caricamento di documenti
# 2. suddivisione in chunk
# 3. creazione di embeddings
# 4. ricerca e recupero (retrieval)
# 5. prompting verso il modello
# 6. con langchain puoi costruire velocemente sistemi RAG su documenti aziendali, PDF, email, database.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Envirnment variables for embeddings model and llm
load_dotenv()
azure_endpoint_embeddings = os.getenv("AZURE_ENDPOINT_EMB")
azure_api_key_embeddings  = os.getenv("AZURE_API_KEY_EMB")

azure_deployment = "gpt-4o"
azure_endpoint_llm = os.getenv("AZURE_ENDPOINT_LLM")
azure_api_key_llm  = os.getenv("AZURE_API_KEY_LLM")

# Input dir
DOCUMENTS_DIR = "documents"

# Text splitter - chuncking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Embeddings client - text-embedding-ada-002
embedding = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint_embeddings,
    api_key=azure_api_key_embeddings,
    deployment="text-embedding-ada-002",
    api_version="2024-12-01-preview"
)

# Llm client - gpt-4o
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint_llm,
    api_key=azure_api_key_llm,
    deployment_name=azure_deployment,
    api_version="2024-12-01-preview",
    temperature=0
)

# Iteration for all docs
for filename in os.listdir(DOCUMENTS_DIR):
    path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.isfile(path):
        continue

    # Data loader    
    loader = TextLoader(path)
    docs = loader.load()

    # Splitter
    split_docs = splitter.split_documents(docs)

    # Vector store db - chromadb
    vectordb = Chroma.from_documents(split_docs, embedding)

    # Retriever
    retriever = vectordb.as_retriever()

    # chain Answer and question
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Question
    query = input('Insert your question: ')

    # Answer
    answer = qa_chain.invoke(query)
    print(f"\n{filename}: {answer}")

