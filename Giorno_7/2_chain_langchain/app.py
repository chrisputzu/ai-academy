import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
azure_endpoint_embeddings = os.getenv("AZURE_ENDPOINT_EMB")
azure_api_key_embeddings  = os.getenv("AZURE_API_KEY_EMB")

azure_deployment = "gpt-4o"
azure_endpoint_llm = os.getenv("AZURE_ENDPOINT_LLM")
azure_api_key_llm = os.getenv("AZURE_API_KEY_LLM")

DOCUMENTS_DIR = "input_folder"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint = azure_endpoint_embeddings,
    api_key = azure_api_key_embeddings,
    deployment = "text-embedding-ada-002",
    api_version = "2024-12-01-preview"
)

llm = AzureChatOpenAI(
    azure_endpoint = azure_endpoint_llm,
    api_key = azure_api_key_llm,
    deployment_name = azure_deployment,
    api_version = "2024-12-01-preview",
    temperature = 0
)


all_chunks = []

for filename in os.listdir(DOCUMENTS_DIR):
    path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.isfile(path):
        docs = TextLoader(path).load()
        for d in docs:
            d.metadata["source"] = filename          
        all_chunks.extend(splitter.split_documents(docs))

vectordb  = Chroma.from_documents(all_chunks, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k":8})  

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    return_source_documents = True
)


chat_history = []

while True:
    query = input("\nInsert your question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"question": query, "chat_history": chat_history})

    print(f"\nAnswer:\n{result['answer']}")

    best_source = result["source_documents"][0].metadata.get("source")

    print("\nSources (only the pertinent file):")
    for i, doc in enumerate(result["source_documents"], 1):
        if doc.metadata.get("source") == best_source:   
            print(f"\n--- Source {i} ---")
            print(f"File Name: {doc.metadata.get('source')}")
            print(f"Content:\n{doc.page_content.strip()}")

    chat_history.append((query, result["answer"]))
