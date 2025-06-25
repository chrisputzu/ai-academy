from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Carica documenti
files = ["Giorno_6/Mail.txt", "Giorno_6/nota.txt", "Giorno_6/Fattura.txt"]
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

# 5. Modello Hugging Face locale (es. TinyLlama)
llm_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 6. Retrieval + QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Domanda
query = "Qual è il nuovo IBAN comunicato da Mario Rossi?"
risposta = qa_chain.invoke({"query": query})

# 8. Output
print("Risultato:", risposta["result"])
print("\nFonte:")
for doc in risposta["source_documents"]:
    print("-", doc.metadata["source"])
