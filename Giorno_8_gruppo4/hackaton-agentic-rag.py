import os
from dotenv import load_dotenv
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline




load_dotenv()  

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TextFileLoader:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Non è una directory: {folder_path}")

    def load(self) -> list[dict]:
        """
        Legge tutti i file .txt nel percorso specificato.
        Ritorna: lista di dizionari con 'file_name' e 'content'
        """
        results = []
        for file in self.folder_path.glob("*.txt"):
            try:
                content = file.read_text(encoding="utf-8")
                results.append({
                    "file_name": file.name,
                    "content": content
                })
            except Exception as e:
                print(f"Errore nella lettura di {file.name}: {e}")
        return results
    

    
class AIProjectClientDefinition:
    def __init__(self):
        self.endpoint = os.getenv("PROJECT_ENDPOINT") 

        self.client = AIProjectClient(
            endpoint=self.endpoint,
            azure_endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )


class AdaEmbeddingModel(AIProjectClientDefinition):
    def __init__(self, client_class: AIProjectClientDefinition, model_name: str = "text-embedding-ada-002"):
        self.client = client_class.client
        self.model_name = model_name
        self.azure_client = client_class.client.inference.get_azure_openai_client(api_version="2023-05-15")


    def embed_text(self, text: str) -> list[float]:
        response = self.azure_client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding



class LangchainAdaWrapper(Embeddings):
    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
    


class TextAnonymizer:
    def __init__(self):
        model_name = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        self.regex_patterns = {
            "IBAN": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
            "CF": r"\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b",
            "PHONE": r"\b(\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{5,8}\b"
        }

        self.entity_mask_map = {
            "PER": "[NOME]",
            "LOC": "[INDIRIZZO]",
            "ORG": "[AZIENDA]",
            "MISC": "[VARIABILE]",
            "IBAN": "[IBAN]",
            "CF": "[CF]",
            "PHONE": "[TELEFONO]"
        }

    def mask_text(self, text: str) -> str:
        masked_text = text

        # Applica regex
        for label, pattern in self.regex_patterns.items():
            matches = re.findall(pattern, masked_text)
            for match in matches:
                masked_text = masked_text.replace(match, self.entity_mask_map.get(label, "[SENSIBILE]"))

        # Applica NER
        entities = self.ner_pipeline(masked_text)
        for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
            original = masked_text[ent['start']:ent['end']]
            label = ent['entity_group']
            if label in self.entity_mask_map:
                masked_text = masked_text[:ent['start']] + self.entity_mask_map[label] + masked_text[ent['end']:]

        return masked_text




class ChatCompletionModel(AIProjectClientDefinition):
    def __init__(self, client_class: AIProjectClientDefinition, model_name: str = "gpt-4o"):
        self.client = client_class.client
        self.model_name = model_name
        self.azure_client = client_class.client.inference.get_azure_openai_client(api_version="2025-01-01-preview")

    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Sei un assistente AI specializzato nell'analisi e nella classificazione di documenti testuali."
                ),
            },
            {
                "role": "user",
                "content": f"Documento:\n{content}\n\nDomanda: {question}"
            }
        ]

        response = self.azure_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content


class RAGPipeline:
    def __init__(self, folder_path: str):
        self.anonymizer = TextAnonymizer()
        loader = TextFileLoader(folder_path)
        results = loader.load()

        ai_client = AIProjectClientDefinition()
        ada_model = AdaEmbeddingModel(ai_client)
        embedding_wrapper = LangchainAdaWrapper(ada_model)

        documents = [
            Document(page_content=doc["content"], metadata={"file_name": doc["file_name"]})
            for doc in results
        ]
        vectorstore = FAISS.from_documents(documents, embedding=embedding_wrapper)
        self.retriever = vectorstore.as_retriever()
        self.chat_model = ChatCompletionModel(ai_client)

    def answer_query(self, query: str) -> str:
        docs_simili = self.retriever.get_relevant_documents(query)
        if not docs_simili:
            return "Nessun documento rilevante trovato."

        risposte = ""
        for doc in docs_simili:
            testo_anonimizzato = self.anonymizer.mask_text(doc.page_content)
            risposta = self.chat_model.ask_about_document(testo_anonimizzato, query)
            risposte += f"**{doc.metadata['file_name']}**\n⚠️ Il contenuto è stato anonimizzato.\n{risposta}\n\n"

        return risposte


if __name__ == "__main__":
    folder_path = r"C:\Users\LA871ZW\OneDrive - EY\Documents\GitHub\Giovanni-Zagaria-ai-academy\Giorno_8\rag_documents"
    pipeline = RAGPipeline(folder_path)

    import gradio as gr
    with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
        gr.Markdown(
            """
            # 📄 RAG con Azure OpenAI
            Interroga i tuoi documenti locali con embeddings **Ada** e risposte da **GPT-4o**.  
            Inserisci una domanda qui sotto e ottieni risposte intelligenti basate sul contenuto.
            """,
            elem_id="title"
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="🧠 Domanda",
                    placeholder="Es. Qual è l’oggetto del documento X?",
                    lines=2,
                    interactive=True
                )
                submit_btn = gr.Button("Cerca tra i documenti", variant="primary")

            with gr.Column(scale=2):
                output_area = gr.Markdown(label="📚 Risposta")

        submit_btn.click(fn=pipeline.answer_query, inputs=query_input, outputs=output_area)

    demo.launch()