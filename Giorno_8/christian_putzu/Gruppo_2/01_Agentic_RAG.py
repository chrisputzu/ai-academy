import streamlit as st
import os
import re
import json
import tempfile
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# LangChain imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai.llm import LLM

from transformers import pipeline
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configurazione centralizzata"""
    NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
    
    # Azure OpenAI Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    # Using separate environment variables for embedding endpoint/key if they differ
    AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_ENDPOINT_EMB")  # For embeddings
    AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_API_KEY_EMB")  # For embeddings
    AZURE_API_VERSION = "2024-02-01" # Using a stable API version for LangChain compatibility
    DEPLOYMENT_NAME = "gpt-4o" # Ensure this matches your chat model deployment name in Azure
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002" # Ensure this matches your embedding model deployment name

# Forza OPENAI_API_KEY nell'ambiente per compatibilità con alcune librerie
if Config.AZURE_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.AZURE_API_KEY

class NERAnonimizer:
    """Classe per anonimizzazione con NER e regex"""
    
    def __init__(self):
        self.regex_patterns = {
            "IBAN": r'\bIT\d{2}[A-Z0-9]{23}\b',
            "EMAIL": r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
            "CF": r'\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b',
            "CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "PHONE": r'\b\+?[0-9\s\-\(\)]{8,15}\b'
        }
        self._ner_pipe = None
    
    @property
    def ner_pipe(self):
        """Lazy loading del modello NER"""
        if self._ner_pipe is None:
            with st.spinner("Caricamento modello NER..."):
                self._ner_pipe = pipeline(
                    "ner",
                    model=Config.NER_MODEL,
                    aggregation_strategy="simple"
                )
        return self._ner_pipe
    
    def mask_with_regex(self, text: str) -> Tuple[str, Dict]:
        """Applica mascheramento con regex"""
        masked_text = text
        found_entities = {}
        
        # Sort patterns by length of regex to prefer longer matches first
        sorted_patterns = sorted(self.regex_patterns.items(), key=lambda item: len(item[1]), reverse=True)

        for label, pattern in sorted_patterns:
            matches = list(re.finditer(pattern, masked_text, flags=re.IGNORECASE))
            for match in reversed(matches):
                original = match.group()
                if original.startswith('[') and original.endswith(']'):
                    continue

                placeholder = f"[{label}_{len(found_entities)}]"
                found_entities[placeholder] = original
                masked_text = masked_text[:match.start()] + placeholder + masked_text[match.end():]
        
        return masked_text, found_entities
    
    def mask_with_ner(self, text: str) -> Tuple[str, Dict]:
        """Applica mascheramento con NER"""
        try:
            entities = self.ner_pipe(text)
            entity_map = {}
            
            sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
            
            for ent in sorted_entities:
                if ent['score'] > 0.5:
                    label = ent['entity_group']
                    original_text = text[ent['start']:ent['end']]
                    
                    if original_text.startswith('[') and original_text.endswith(']'):
                        continue

                    placeholder = f"[{label}_{len(entity_map)}]"
                    entity_map[placeholder] = original_text
                    
                    text = text[:ent['start']] + placeholder + text[ent['end']:]
            
            return text, entity_map
            
        except Exception as e:
            st.error(f"Errore NER: {e}")
            return text, {}
    
    def anonymize(self, text: str) -> Tuple[str, Dict]:
        """Pipeline completa di anonimizzazione"""
        masked_text, regex_entities = self.mask_with_regex(text)
        final_text, ner_entities = self.mask_with_ner(masked_text)
        
        all_entities = {**regex_entities, **ner_entities}
        
        return final_text, all_entities

class AzureProcessor:
    """Processore Azure OpenAI"""
    
    def __init__(self):
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup client Azure"""
        if Config.AZURE_API_KEY and Config.AZURE_ENDPOINT:
            try:
                self.client = AzureOpenAI(
                    api_key=Config.AZURE_API_KEY,
                    api_version=Config.AZURE_API_VERSION,
                    azure_endpoint=Config.AZURE_ENDPOINT
                )
            except Exception as e:
                st.error(f"Errore configurazione Azure OpenAI: {e}")
                self.client = None
        else:
            st.warning("Credenziali Azure OpenAI non trovate. L'analisi AI non sarà disponibile.")
    
    def process_document(self, anonymized_text: str) -> str:
        """Processa documento con AI"""
        if not self.client:
            return "Azure OpenAI non configurato. Configura le credenziali nelle variabili d'ambiente."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Sei un assistente intelligente che riceve documenti anonimizzati e svolge le seguenti operazioni:\n\n"
                        "1. Fornisci il tipo di documento (es: email, fattura, contratto, altro).\n"
                        "2. Fornisci un riepilogo chiaro e conciso del contenuto (max 5 righe).\n"
                        "3. Esegui un'analisi semantica del testo, evidenziando i temi principali, sentimenti, e intenzioni implicite.\n"
                        "4. Se il documento è una comunicazione da o verso un cliente, genera una possibile risposta adeguata, professionale e contestuale.\n"
                        "5. Lavora solo con i contenuti presenti nel documento anonimizzato, evitando ogni supposizione che possa violare la privacy.\n\n"
                        "6. Rispondi solo in italiano."
                        "Formatta la risposta in modo chiaro con sezioni separate."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analizza questo documento:\n\n{anonymized_text}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model=Config.DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Errore durante l'analisi AI: {e}"

class RAGChatbot:
    """Chatbot basato su RAG per interrogare documenti anonimizzati utilizzando LangChain."""
 
    def __init__(self):
        self.vector_store = None # Will store FAISS index
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.setup_langchain_components()
 
    def setup_langchain_components(self):
        """Setup LangChain components (embeddings and LLM)."""
        # Check if both main API key/endpoint and embedding API key/endpoint are provided
        if not (Config.AZURE_API_KEY and Config.AZURE_ENDPOINT and
                Config.AZURE_EMBEDDING_API_KEY and Config.AZURE_EMBEDDING_ENDPOINT):
            st.warning("Credenziali Azure OpenAI (chat o embedding) non trovate. Il chatbot RAG non sarà disponibile.")
            return
 
        try:
            # Initialize Azure OpenAI Embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                model=Config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                api_version=Config.AZURE_API_VERSION,
                azure_endpoint=Config.AZURE_EMBEDDING_ENDPOINT,
                api_key=Config.AZURE_EMBEDDING_API_KEY,
                chunk_size=16 # Recommended chunk size for Azure embeddings
            )
            
            # Initialize Azure Chat OpenAI LLM - FIXED
            self.llm = AzureChatOpenAI(
                deployment_name=Config.DEPLOYMENT_NAME,
                azure_endpoint=Config.AZURE_ENDPOINT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                temperature=0.2
            )
        except Exception as e:
            st.error(f"Errore configurazione LangChain Azure OpenAI: {e}")
            self.embeddings = None
            self.llm = None
 
    def build_vector_store(self, anonymized_docs: Dict[str, Dict]):
        """
        Builds the FAISS vector store from anonymized documents using LangChain.
        """
        if not self.embeddings or not self.llm:
            st.error("Componenti LangChain non configurati. Impossibile costruire il knowledge base.")
            return
 
        all_texts_for_rag = []
        for filename, doc_data in anonymized_docs.items():
            if doc_data.get('confirmed', False):
                # Add filename as metadata or prefix to content for better context
                all_texts_for_rag.append(f"Documento {filename}:\n{doc_data['anonymized']}")
 
        if not all_texts_for_rag:
            st.warning("Nessun documento confermato per costruire il knowledge base.")
            return
 
        with st.spinner("Creando chunks e embeddings con LangChain..."):
            # Combine all texts into a single string for splitting
            combined_text = "\n\n".join(all_texts_for_rag)
 
            # Initialize LangChain's text splitter
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000, # Adjust chunk size as needed
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_text(combined_text)
            # Create FAISS index from texts and embeddings
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            st.success(f"Vector store costruito con {len(texts)} chunks.")
            
            # Define a custom prompt for the QA chain
            qa_prompt_template = """Usa i seguenti pezzi di contesto per rispondere alla domanda.
            Se non conosci la risposta, dì semplicemente che non la sai, non cercare di inventare una risposta.
 
            {context}
 
            Domanda: {question}
            Risposta utile:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt_template)
 
            # Initialize RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", # "stuff" is good for smaller contexts, "map_reduce" for larger
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True, # Set to True to see which documents were used
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt
            )
 
    def answer_question(self, query: str) -> str:
        """Generates an answer using LangChain's RetrievalQA chain."""
        if not self.qa_chain:
            return "Chatbot non pronto. Assicurati che il knowledge base sia stato costruito."
 
        try:
            # Use 'query' as the key for the invoke method
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]
            # Optionally, display source documents
            source_docs = result.get("source_documents", [])
            if source_docs:
                answer += "\n\n**Fonti:**\n"
                for i, doc in enumerate(source_docs):
                    # Extract filename if it was embedded in the chunk
                    match = re.search(r"Documento (.*?):\n", doc.page_content)
                    filename_info = f" (da {match.group(1)})" if match else ""
                    answer += f"- ...{doc.page_content[-100:]}{filename_info}\n" # Show last 100 chars of chunk
            return answer
        except Exception as e:
            return f"Errore durante la generazione della risposta AI con LangChain: {e}"
    
    def get_relevant_context(self, query: str, max_docs: int = 3) -> str:
        """Estrae il contesto rilevante per una query senza generare una risposta completa"""
        if not self.vector_store:
            return ""
        
        try:
            docs = self.vector_store.similarity_search(query, k=max_docs)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Errore nell'estrazione del contesto: {e}"

class CrewAIManager:
    """Gestione degli agenti CrewAI"""
   
    def __init__(self, rag_chatbot: RAGChatbot):
        self.rag_chatbot = rag_chatbot
        self.agents = None
        self.llm = None
        self.setup_crew()
   
    def setup_crew(self):
        """Setup del crew CrewAI"""
        if not Config.AZURE_API_KEY:
            st.warning("Credenziali Azure non disponibili per CrewAI")
            return
       
        try:
            # LLM per CrewAI con configurazione Azure
            self.llm = LLM(
                model=f"azure/{Config.DEPLOYMENT_NAME}",
                api_key=Config.AZURE_API_KEY,
                base_url=Config.AZURE_ENDPOINT,
                api_version=Config.AZURE_API_VERSION
            )
           
            # Agente per analisi documenti
            document_analyst = Agent(
                role="Senior Document Analyst",
                goal="Analizzare documenti anonimizzati e fornire insights approfonditi sui contenuti, "
                     "identificando pattern, tematiche ricorrenti e elementi di interesse aziendale",
                backstory="Sei un esperto analista di documenti con oltre 15 anni di esperienza in privacy, "
                         "compliance e analisi semantica. Hai una profonda conoscenza delle normative GDPR "
                         "e lavori esclusivamente con documenti anonimizzati per garantire la massima protezione dei dati. "
                         "La tua specialità è estrarre insights di business mantenendo la conformità normativa.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            # Agente per RAG queries
            rag_specialist = Agent(
                role="RAG Query Specialist",
                goal="Rispondere a domande complesse utilizzando il sistema RAG per recuperare "
                     "informazioni precise dai documenti anonimizzati e fornire risposte contestualizzate",
                backstory="Sei un esperto in sistemi di Information Retrieval e Retrieval-Augmented Generation "
                         "con specializzazione in elaborazione di documenti enterprise. Hai sviluppato sistemi RAG "
                         "per Fortune 500 companies e sei esperto nell'ottimizzazione di query semantiche "
                         "su grandi collezioni documentali anonimizzate.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            # Agente coordinatore e strategista
            strategy_coordinator = Agent(
                role="Strategic Analysis Coordinator",
                goal="Coordinare l'analisi multi-agente, sintetizzare i risultati e fornire "
                     "raccomandazioni strategiche actionable basate sui documenti analizzati",
                backstory="Sei un senior consultant con background in strategic management e data analysis. "
                         "Hai guidato progetti di trasformazione digitale per aziende multinazionali "
                         "e sei esperto nel tradurre insights tecnici in raccomandazioni business concrete. "
                         "La tua forza è coordinare team multidisciplinari e fornire sintesi executive-ready.",
                llm=self.llm,
                verbose=True,
                allow_delegation=True,
                max_iter=4
            )
           
            # Agente per sentiment e trend analysis
            sentiment_analyst = Agent(
                role="Sentiment & Trend Analyst",
                goal="Analizzare il sentiment, le emozioni e i trend emergenti nei documenti "
                     "per identificare opportunità e rischi nascosti",
                backstory="Sei un esperto in sentiment analysis e behavioral analytics con PhD in Psychology "
                         "e specializzazione in NLP. Hai sviluppato modelli predittivi per analizzare "
                         "comunicazioni aziendali e identificare early warning signals. "
                         "La tua expertise include analisi del linguaggio, emotion detection e trend forecasting.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            self.agents = {
                'document_analyst': document_analyst,
                'rag_specialist': rag_specialist,
                'strategy_coordinator': strategy_coordinator,
                'sentiment_analyst': sentiment_analyst
            }
           
            st.success("✅ Agenti CrewAI configurati con successo")
           
        except Exception as e:
            st.error(f"Errore setup CrewAI: {e}")
            self.agents = None
   
    def create_comprehensive_analysis_task(self, query: str, analysis_type: str = "comprehensive") -> str:
        """Crea task di analisi comprensiva per il crew"""
        if not self.agents:
            return "CrewAI non configurato correttamente"
       
        try:
            # Ottieni contesto rilevante dal RAG
            context = self.rag_chatbot.get_relevant_context(query, max_docs=5)
           
            tasks = []
           
            if analysis_type in ["comprehensive", "document"]:
                # Task di analisi documentale
                doc_analysis_task = Task(
                    description=f"""
                    Analizza questi documenti per: {query}
                   
                    CONTESTO: {context}
                   
                    Fornisci:
                    - Tipo e classificazione dei documenti
                    - Temi e argomenti principali
                    - Elementi rilevanti per il business
                    - Note su compliance e conformità
                    """,
                    expected_output="Analisi strutturata con classificazione e insights",
                    agent=self.agents['document_analyst']
                )
                tasks.append(doc_analysis_task)
           
            if analysis_type in ["comprehensive", "sentiment"]:
                # Task di sentiment analysis
                sentiment_task = Task(
                    description=f"""
                    Analizza sentiment e emozioni per: {query}
                   
                    CONTESTO: {context}
                   
                    Valuta:
                    - Sentiment generale (scala 1-10)
                    - Emozioni prevalenti
                    - Trend nelle comunicazioni
                    - Segnali di rischio o opportunità
                    """,
                    expected_output="Analisi sentiment con valutazioni quantitative",
                    agent=self.agents['sentiment_analyst']
                )
                tasks.append(sentiment_task)
           
            if analysis_type in ["comprehensive", "rag"]:
                # Task di query RAG specializzata
                rag_task = Task(
                    description=f"""
                    Rispondi usando RAG a: {query}
                   
                    CONTESTO: {context}
                   
                    Includi:
                    - Risposta diretta alla query
                    - Evidenze dai documenti
                    - Correlazioni trovate
                    - Informazioni mancanti
                    - Suggerimenti per approfondire
                    """,
                    expected_output="Risposta RAG con evidenze e correlazioni",
                    agent=self.agents['rag_specialist']
                )
                tasks.append(rag_task)
           
            # Task di coordinamento strategico (sempre incluso)
            coordination_task = Task(
                description=f"""
                Sintetizza tutti i risultati per: {query}
               
                Crea sintesi con:
                - Executive Summary (3 punti chiave)
                - Insights strategici
                - Raccomandazioni prioritarie
                - Next steps concreti
                - Valutazione rischi
               
                Output executive-ready e orientato all'azione.
                """,
                expected_output="Sintesi strategica con raccomandazioni actionable",
                agent=self.agents['strategy_coordinator']
            )
            tasks.append(coordination_task)
           
            # Crea crew per questa analisi
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                verbose=True
            )
           
            with st.spinner(f"Eseguendo analisi {analysis_type} con {len(tasks)} agenti CrewAI..."):
                result = crew.kickoff()
           
            return str(result)
           
        except Exception as e:
            return f"Errore nell'analisi CrewAI: {e}"
    
    def create_custom_task(self, query: str, selected_agents: List[str], custom_instructions: str = "") -> str:
        """Crea task personalizzate con agenti specifici"""
        if not self.agents:
            return "CrewAI non configurato correttamente"
        
        try:
            context = self.rag_chatbot.get_relevant_context(query, max_docs=5)
            
            tasks = []
            agents_to_use = []
            
            # Seleziona gli agenti richiesti
            for agent_key in selected_agents:
                if agent_key in self.agents:
                    agents_to_use.append(self.agents[agent_key])
                    
                    # Crea task specifico per ogni agente
                    task = Task(
                        description=f"""
                        {custom_instructions}
                        
                        QUERY: {query}
                        CONTESTO: {context}
                        
                        Esegui l'analisi secondo il tuo ruolo specifico e le istruzioni personalizzate fornite.
                        """,
                        expected_output=f"Analisi specializzata secondo il ruolo di {agent_key}",
                        agent=self.agents[agent_key]
                    )
                    tasks.append(task)
            
            if not tasks:
                return "Nessun agente valido selezionato"
            
            crew = Crew(
                agents=agents_to_use,
                tasks=tasks,
                verbose=True
            )
            
            with st.spinner(f"Eseguendo task personalizzata con {len(agents_to_use)} agenti..."):
                result = crew.kickoff()
            
            return str(result)
            
        except Exception as e:
            return f"Errore nel task personalizzato CrewAI: {e}"

def init_session_state():
    """Inizializza lo stato della sessione"""
    if 'anonymizer' not in st.session_state:
        st.session_state.anonymizer = NERAnonimizer()
    if 'processor' not in st.session_state:
        st.session_state.processor = AzureProcessor()
    if 'rag_chatbot' not in st.session_state:
        st.session_state.rag_chatbot = RAGChatbot()
    if 'crewai_manager' not in st.session_state:
        st.session_state.crewai_manager = CrewAIManager(st.session_state.rag_chatbot)
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'anonymized_docs' not in st.session_state:
        st.session_state.anonymized_docs = {}
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'crewai_history' not in st.session_state:
        st.session_state.crewai_history = []
    if 'vector_store_built' not in st.session_state:
        st.session_state.vector_store_built = False

def display_entity_editor(entities: Dict, doc_key: str):
    """Mostra editor per le entità rilevate"""
    if not entities:
        st.info("Nessuna entità sensibile rilevata.")
        return entities
    
    st.subheader("🔍 Entità rilevate")
    st.write("Verifica e modifica le entità sensibili rilevate:")
    
    # Use a temporary list to manage entities for display and modification
    current_entities_list = list(entities.items())
    updated_entities_dict = {}
    
    # Keep track of which entities are marked for deletion
    deleted_placeholders = set()

    for i, (placeholder, original_value) in enumerate(current_entities_list):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.write(f"**{placeholder}**")
        
        with col2:
            new_value = st.text_input(
                f"Valore originale",
                value=original_value,
                key=f"{doc_key}_{placeholder}_value_{i}" # Unique key for each widget
            )
            updated_entities_dict[placeholder] = new_value
        
        with col3:
            # Use a unique key for the button
            if st.button("🗑️", key=f"{doc_key}_{placeholder}_delete_{i}", help="Rimuovi questa entità"):
                deleted_placeholders.add(placeholder)
                # No need to rerun here, the rerun will happen after the loop
    
    # After the loop, apply deletions and update the session state
    if deleted_placeholders:
        # Reconstruct the entities dictionary without the deleted ones
        final_entities = {k: v for k, v in updated_entities_dict.items() if k not in deleted_placeholders}
        st.session_state.anonymized_docs[doc_key]['entities'] = final_entities
        
        # Re-anonymize the original text to reflect the removed entities
        # This will revert any manual edits to the 'anonymized_text' field
        st.session_state.anonymized_docs[doc_key]['anonymized'], _ = st.session_state.anonymizer.anonymize(
            st.session_state.anonymized_docs[doc_key]['original']
        )
        st.session_state.vector_store_built = False # Invalidate vector store
        st.rerun() # Rerun to update the UI with the changes
    
    return updated_entities_dict # Return the potentially modified entities for confirmation

def main():
    st.set_page_config(
        page_title="Anonimizzatore Documenti",
        page_icon="🔒",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🔒 Anonimizzatore Documenti con NER, RAG e CrewAI")
    st.markdown("---")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Status Azure
        if Config.AZURE_API_KEY and Config.AZURE_ENDPOINT:
            st.success("✅ Azure OpenAI configurato")
            st.info(f"Chat Model: {Config.DEPLOYMENT_NAME}")
            st.info(f"Embedding Model: {Config.AZURE_EMBEDDING_DEPLOYMENT_NAME}")
            
            # Status CrewAI
            if st.session_state.crewai_manager.agents:
                st.success("✅ CrewAI Agenti attivi")
                with st.expander("👥 Agenti Disponibili"):
                    for agent_key, agent in st.session_state.crewai_manager.agents.items():
                        st.write(f"**{agent.role}**")
                        st.write(f"_{agent.goal}_")
            else:
                st.warning("⚠️ CrewAI non configurato")
        else:
            st.error("❌ Azure OpenAI non configurato")
            st.write("Configura le variabili d'ambiente:")
            st.code("""
AZURE_ENDPOINT=your_endpoint
AZURE_API_KEY=your_api_key
AZURE_ENDPOINT_EMB=your_embedding_endpoint
AZURE_API_KEY_EMB=your_embedding_api_key
DEPLOYMENT_NAME=your_chat_deployment_name
AZURE_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
            """)
        
        st.markdown("---")
        
        # Statistiche documenti
        if st.session_state.uploaded_files:
            st.subheader("📊 Statistiche")
            uploaded_count = len(st.session_state.uploaded_files)
            anonymized_count = len(st.session_state.anonymized_docs)
            confirmed_count = sum(1 for doc in st.session_state.anonymized_docs.values() if doc.get('confirmed', False))
            
            st.metric("File caricati", uploaded_count)
            st.metric("Anonimizzati", anonymized_count)
            st.metric("Confermati", confirmed_count)
            
            if confirmed_count > 0:
                if st.session_state.vector_store_built:
                    st.success("✅ Knowledge Base pronto")
                else:
                    st.info("🔄 Knowledge Base da costruire")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset sessione"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload", "🔍 Anonimizzazione", "📊 Analisi", "💬 Chatbot RAG", "🤖 CrewAI"])
    
    with tab1:
        st.header("📤 Carica Documenti")
        
        uploaded_files = st.file_uploader(
            "Carica uno o più file .txt",
            type=['txt'],
            accept_multiple_files=True,
            help="Seleziona i file di testo da anonimizzare"
        )
        
        if uploaded_files:
            new_files_uploaded = False
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    content = file.read().decode('utf-8')
                    st.session_state.uploaded_files[file.name] = {
                        'content': content,
                        'size': len(content)
                    }
                    new_files_uploaded = True
            
            if new_files_uploaded:
                st.success(f"Caricati {len(uploaded_files)} file")
                # Reset anonymized/processed docs if new files are uploaded
                st.session_state.anonymized_docs = {}
                st.session_state.processed_docs = {}
                st.session_state.vector_store_built = False
                st.session_state.chat_history = []
                st.session_state.crewai_history = []
                st.rerun() # Rerun to clear previous state related to old files
            else:
                st.info("Nessun nuovo file caricato.")
            
            # Mostra anteprima file caricati
            st.subheader("📄 File caricati")
            for filename, file_data in st.session_state.uploaded_files.items():
                with st.expander(f"📄 {filename} ({file_data['size']} caratteri)"):
                    st.text_area(
                        "Contenuto originale",
                        value=file_data['content'][:500] + ("..." if len(file_data['content']) > 500 else ""),
                        height=150,
                        disabled=True,
                        key=f"preview_{filename}"
                    )
    
    with tab2:
        st.header("🔍 Anonimizzazione e Revisione")
        
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Carica prima alcuni documenti nella tab 'Upload'")
        else:
            # Bottone per avviare anonimizzazione
            if st.button("🚀 Avvia Anonimizzazione", type="primary"):
                progress_bar = st.progress(0)
                total_files = len(st.session_state.uploaded_files)
                
                for i, (filename, file_data) in enumerate(st.session_state.uploaded_files.items()):
                    progress_bar.progress((i + 1) / total_files, f"Processando {filename}...")
                    
                    # Applica anonimizzazione
                    anonymized_text, entities = st.session_state.anonymizer.anonymize(file_data['content'])
                    
                    st.session_state.anonymized_docs[filename] = {
                        'original': file_data['content'],
                        'anonymized': anonymized_text,
                        'entities': entities,
                        'confirmed': False
                    }
                
                progress_bar.empty()
                st.success("✅ Anonimizzazione completata! Revisiona le entità e conferma.")
                st.session_state.vector_store_built = False # Invalidate vector store on re-anonymization
                st.rerun()
            
            # Mostra documenti anonimizzati per revisione
            if st.session_state.anonymized_docs:
                st.subheader("📝 Revisiona Documenti Anonimizzati")
                
                for filename, doc_data in st.session_state.anonymized_docs.items():
                    with st.expander(f"📄 {filename} {'✅' if doc_data['confirmed'] else '⏳'}", expanded=not doc_data['confirmed']):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Testo Originale:**")
                            st.text_area(
                                "Originale",
                                value=doc_data['original'][:300] + ("..." if len(doc_data['original']) > 300 else ""),
                                height=200,
                                disabled=True,
                                key=f"orig_{filename}",
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            st.write("**Testo Anonimizzato:**")
                            
                            # Editor per il testo anonimizzato
                            edited_text = st.text_area(
                                "Anonimizzato (modificabile)",
                                value=doc_data['anonymized'],
                                height=200,
                                key=f"anon_{filename}",
                                label_visibility="collapsed"
                            )
                            
                            # Aggiorna il testo se modificato
                            if edited_text != doc_data['anonymized']:
                                st.session_state.anonymized_docs[filename]['anonymized'] = edited_text
                        
                        # Editor delle entità
                        # Pass a copy of entities to avoid modifying dictionary during iteration if reran
                        updated_entities_from_editor = display_entity_editor(dict(doc_data['entities']), filename)
                        
                        # Bottoni di azione
                        col_confirm, col_reset = st.columns(2)
                        
                        with col_confirm:
                            if st.button(f"✅ Conferma {filename}", key=f"confirm_{filename}"):
                                st.session_state.anonymized_docs[filename]['confirmed'] = True
                                st.session_state.anonymized_docs[filename]['entities'] = updated_entities_from_editor
                                st.success(f"✅ {filename} confermato!")
                                st.session_state.vector_store_built = False # Invalidate vector store
                                st.rerun()
                        
                        with col_reset:
                            if st.button(f"🔄 Reset {filename}", key=f"reset_{filename}"):
                                # Reset alle impostazioni originali
                                original_data = st.session_state.uploaded_files[filename]
                                anonymized_text, entities = st.session_state.anonymizer.anonymize(original_data['content'])
                                
                                st.session_state.anonymized_docs[filename] = {
                                    'original': original_data['content'],
                                    'anonymized': anonymized_text,
                                    'entities': entities,
                                    'confirmed': False
                                }
                                st.session_state.vector_store_built = False # Invalidate vector store
                                st.rerun()
                
                # Statistiche
                confirmed_count = sum(1 for doc in st.session_state.anonymized_docs.values() if doc['confirmed'])
                total_count = len(st.session_state.anonymized_docs)
                
                st.metric(
                    "Progresso Conferme",
                    f"{confirmed_count}/{total_count}",
                    delta=f"{(confirmed_count/total_count)*100:.1f}%" if total_count > 0 else "0%"
                )
    
    with tab3:
        st.header("📊 Analisi AI")
        
        confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() if v.get('confirmed', False)}
        
        if not confirmed_docs:
            st.warning("⚠️ Conferma prima alcuni documenti anonimizzati nella tab 'Anonimizzazione'")
        else:
            st.write(f"Documenti confermati pronti per l'analisi: **{len(confirmed_docs)}**")
            
            if st.button("🤖 Avvia Analisi AI", type="primary"):
                progress_bar = st.progress(0)
                
                for i, (filename, doc_data) in enumerate(confirmed_docs.items()):
                    progress_bar.progress((i + 1) / len(confirmed_docs), f"Analizzando {filename}...")
                    
                    # Processa con Azure AI
                    analysis = st.session_state.processor.process_document(doc_data['anonymized'])
                    
                    st.session_state.processed_docs[filename] = {
                        'anonymized_text': doc_data['anonymized'],
                        'entities_count': len(doc_data['entities']),
                        'analysis': analysis,
                        'entities': doc_data['entities']
                    }
                
                progress_bar.empty()
                st.success("✅ Analisi completata!")
            
            # Mostra risultati analisi
            if st.session_state.processed_docs:
                st.subheader("📋 Risultati Analisi")
                
                for filename, result in st.session_state.processed_docs.items():
                    with st.expander(f"📊 Analisi: {filename}"):
                        
                        # Metriche
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Caratteri testo", len(result['anonymized_text']))
                        col2.metric("Entità trovate", result['entities_count'])
                        col3.metric("Stato", "✅ Completato")
                        
                        # Testo anonimizzato
                        st.subheader("📄 Testo Anonimizzato")
                        st.text_area(
                            "Testo processato",
                            value=result['anonymized_text'],
                            height=150,
                            disabled=True,
                            key=f"analysis_text_{filename}"
                        )
                        
                        # Analisi AI
                        st.subheader("🤖 Analisi AI")
                        st.markdown(result['analysis'])
                        
                        # Entità (se presenti)
                        if result['entities']:
                            st.subheader("🔍 Entità Anonimizzate")
                            entities_df = pd.DataFrame([
                                {'Placeholder': k, 'Valore Originale': v, 'Tipo': k.split('_')[0].replace('[', '')}
                                for k, v in result['entities'].items()
                            ])
                            st.dataframe(entities_df, use_container_width=True)
                        
                        # Download dei risultati
                        result_json = json.dumps({
                            'filename': filename,
                            'anonymized_text': result['anonymized_text'],
                            'analysis': result['analysis'],
                            'entities': result['entities'],
                            'metadata': {
                                'processed_at': str(pd.Timestamp.now()),
                                'entities_count': result['entities_count'],
                                'text_length': len(result['anonymized_text'])
                            }
                        }, indent=2, ensure_ascii=False)
                        
                        st.download_button(
                            label=f"💾 Scarica risultati {filename}",
                            data=result_json,
                            file_name=f"analisi_{filename}.json",
                            mime="application/json",
                            key=f"download_{filename}"
                        )

    with tab4:
        st.header("💬 Chatta con i Documenti")

        confirmed_docs_for_rag = {k: v for k, v in st.session_state.anonymized_docs.items() if v.get('confirmed', False)}

        if not confirmed_docs_for_rag:
            st.warning("⚠️ Carica e conferma prima alcuni documenti anonimizzati nelle tab 'Upload' e 'Anonimizzazione' per abilitare il chatbot.")
        else:
            if not st.session_state.vector_store_built:
                with st.spinner("Costruendo il knowledge base per il chatbot..."):
                    st.session_state.rag_chatbot.build_vector_store(confirmed_docs_for_rag)
                    st.session_state.vector_store_built = True
            
            # Check if vector store was successfully built
            if st.session_state.rag_chatbot.vector_store:
                st.info(f"Chatbot abilitato per {len(confirmed_docs_for_rag)} documenti confermati.")

                # Display chat messages from history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Accept user input
                if prompt := st.chat_input("Fai una domanda sui documenti..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Cercando e generando risposta..."):
                            # Generate answer using LangChain RAG
                            response = st.session_state.rag_chatbot.answer_question(prompt)
                            st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.error("Impossibile costruire il knowledge base. Controlla le credenziali Azure e i log.")

    with tab5:
        st.header("🤖 Analisi Multi-Agente CrewAI")
        
        confirmed_docs_for_crewai = {k: v for k, v in st.session_state.anonymized_docs.items() if v.get('confirmed', False)}
        
        if not confirmed_docs_for_crewai:
            st.warning("⚠️ Conferma prima alcuni documenti anonimizzati per abilitare l'analisi CrewAI.")
        elif not st.session_state.crewai_manager.agents:
            st.error("❌ CrewAI non configurato. Verifica le credenziali Azure OpenAI.")
        else:
            # Assicurati che il vector store sia costruito
            if not st.session_state.vector_store_built:
                with st.spinner("Preparando knowledge base per CrewAI..."):
                    st.session_state.rag_chatbot.build_vector_store(confirmed_docs_for_crewai)
                    st.session_state.vector_store_built = True
            
            st.success(f"🎯 CrewAI pronto per analizzare {len(confirmed_docs_for_crewai)} documenti")
            
            # Sezione configurazione analisi
            st.subheader("⚙️ Configurazione Analisi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_type = st.selectbox(
                    "Tipo di Analisi",
                    options=["comprehensive", "document", "sentiment", "rag", "custom"],
                    format_func=lambda x: {
                        "comprehensive": "🔍 Analisi Comprensiva",
                        "document": "📄 Analisi Documentale",
                        "sentiment": "😊 Sentiment Analysis",
                        "rag": "🔍 Query RAG Avanzata",
                        "custom": "⚙️ Personalizzata"
                    }[x],
                    help="Seleziona il tipo di analisi da eseguire"
                )
            
            with col2:
                if analysis_type == "custom":
                    selected_agents = st.multiselect(
                        "Agenti da utilizzare",
                        options=list(st.session_state.crewai_manager.agents.keys()),
                        default=["strategy_coordinator"],
                        format_func=lambda x: {
                            "document_analyst": "📄 Document Analyst",
                            "rag_specialist": "🔍 RAG Specialist", 
                            "strategy_coordinator": "🎯 Strategy Coordinator",
                            "sentiment_analyst": "😊 Sentiment Analyst"
                        }.get(x, x)
                    )
                else:
                    selected_agents = []
            
            # Input query
            st.subheader("❓ Query per l'Analisi")
            query_input = st.text_area(
                "Inserisci la tua domanda o richiesta di analisi:",
                placeholder="Es: Analizza i temi principali nei documenti e identifica possibili rischi operativi...",
                height=100
            )
            
            # Istruzioni personalizzate per analisi custom
            if analysis_type == "custom":
                custom_instructions = st.text_area(
                    "Istruzioni Personalizzate (opzionale):",
                    placeholder="Fornisci istruzioni specifiche per gli agenti selezionati...",
                    height=80
                )
            else:
                custom_instructions = ""
            
            # Bottone per avviare analisi
            col_analyze, col_clear = st.columns(2)
            
            with col_analyze:
                if st.button("🚀 Avvia Analisi CrewAI", type="primary", disabled=not query_input.strip()):
                    if analysis_type == "custom" and not selected_agents:
                        st.error("Seleziona almeno un agente per l'analisi personalizzata")
                    else:
                        # Esegui analisi
                        with st.spinner(f"Eseguendo analisi {analysis_type} con CrewAI..."):
                            if analysis_type == "custom":
                                result = st.session_state.crewai_manager.create_custom_task(
                                    query_input, selected_agents, custom_instructions
                                )
                            else:
                                result = st.session_state.crewai_manager.create_comprehensive_analysis_task(
                                    query_input, analysis_type
                                )
                        
                        # Salva nei risultati
                        analysis_result = {
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": query_input,
                            "analysis_type": analysis_type,
                            "result": result,
                            "agents_used": selected_agents if analysis_type == "custom" else "auto"
                        }
                        
                        st.session_state.crewai_history.append(analysis_result)
                        st.success("✅ Analisi CrewAI completata!")
            
            with col_clear:
                if st.button("🗑️ Pulisci Cronologia"):
                    st.session_state.crewai_history = []
                    st.success("Cronologia pulita!")
                    st.rerun()
            
            # Mostra risultati delle analisi
            if st.session_state.crewai_history:
                st.subheader("📋 Risultati Analisi CrewAI")
                
                # Mostra risultati in ordine inverso (più recenti prima)
                for i, analysis in enumerate(reversed(st.session_state.crewai_history)):
                    with st.expander(
                        f"🤖 Analisi {len(st.session_state.crewai_history)-i}: {analysis['analysis_type'].upper()} - {analysis['timestamp']}"
                    ):
                        # Header con informazioni
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            st.metric("Tipo Analisi", analysis['analysis_type'].capitalize())
                        
                        with col_info2:
                            st.metric("Timestamp", analysis['timestamp'])
                        
                        with col_info3:
                            agents_used = analysis.get('agents_used', 'auto')
                            if agents_used == 'auto':
                                agent_count = "Automatico"
                            elif isinstance(agents_used, list):
                                agent_count = f"{len(agents_used)} agenti"
                            else:
                                agent_count = str(agents_used)
                            st.metric("Agenti", agent_count)
                        
                        # Query originale
                        st.subheader("❓ Query Originale")
                        st.info(analysis['query'])
                        
                        # Risultato dell'analisi
                        st.subheader("🎯 Risultato Analisi")
                        st.markdown(analysis['result'])
                        
                        # Download del risultato
                        result_json = json.dumps(analysis, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            label="💾 Scarica Risultato",
                            data=result_json,
                            file_name=f"crewai_analysis_{analysis['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                            mime="application/json",
                            key=f"download_crewai_{i}"
                        )
            
            # Sezione esempi di query
            with st.expander("💡 Esempi di Query per CrewAI"):
                st.markdown("""
                **Analisi Comprensiva:**
                - "Fornisci un'analisi completa dei documenti identificando rischi, opportunità e raccomandazioni strategiche"
                - "Analizza la comunicazione aziendale e suggerisci miglioramenti nella gestione clienti"
                
                **Analisi Documentale:**
                - "Classifica i documenti per tipologia e identifica pattern ricorrenti"
                - "Analizza la struttura e organizzazione delle informazioni nei documenti"
                
                **Sentiment Analysis:**
                - "Valuta il sentiment generale nelle comunicazioni e identifica aree di miglioramento"
                - "Analizza le emozioni e i trend nei feedback dei clienti"
                
                **Query RAG Avanzata:**
                - "Trova tutte le menzioni di problemi operativi e le relative soluzioni proposte"
                - "Estrai informazioni su scadenze, deadline e milestone importanti"
                
                **Personalizzata:**
                - Combina agenti specifici per analisi mirate alle tue esigenze
                """)


if __name__ == "__main__":
    main()