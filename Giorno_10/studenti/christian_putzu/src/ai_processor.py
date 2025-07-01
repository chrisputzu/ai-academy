"""
Tutti i componenti AI: Azure, RAG e CrewAI.
"""

import re
from typing import Dict, List
import streamlit as st
from openai import AzureOpenAI

# LangChain imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai.llm import LLM

from config import Config

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
                st.error(f"Errore Azure OpenAI: {e}")
                self.client = None
        else:
            st.warning("Credenziali Azure OpenAI non trovate.")
    
    def process_document(self, anonymized_text: str) -> str:
        """Processa documento con AI"""
        if not self.client:
            return "Azure OpenAI non configurato."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Analizza il documento anonimizzato e fornisci:\n"
                        "1. Tipo di documento\n"
                        "2. Riepilogo (max 5 righe)\n"
                        "3. Analisi semantica (temi, sentiment)\n"
                        "4. Risposta suggerita se è comunicazione cliente\n"
                        "Usa solo i contenuti del documento fornito."
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
            return f"Errore analisi AI: {e}"

class RAGChatbot:
    """Chatbot RAG con LangChain"""
 
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.setup_langchain_components()
 
    def setup_langchain_components(self):
        """Setup componenti LangChain"""
        if not (Config.AZURE_API_KEY and Config.AZURE_ENDPOINT and
                Config.AZURE_EMBEDDING_API_KEY and Config.AZURE_EMBEDDING_ENDPOINT):
            st.warning("Credenziali Azure incomplete. RAG non disponibile.")
            return
 
        try:
            # Embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                model=Config.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                api_version=Config.AZURE_API_VERSION,
                azure_endpoint=Config.AZURE_EMBEDDING_ENDPOINT,
                api_key=Config.AZURE_EMBEDDING_API_KEY,
                chunk_size=16
            )
            
            # LLM
            self.llm = AzureChatOpenAI(
                deployment_name=Config.DEPLOYMENT_NAME,
                azure_endpoint=Config.AZURE_ENDPOINT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                temperature=0.2
            )
        except Exception as e:
            st.error(f"Errore setup LangChain: {e}")
            self.embeddings = None
            self.llm = None
 
    def build_vector_store(self, anonymized_docs: Dict[str, Dict]):
        """Costruisce vector store FAISS"""
        if not self.embeddings or not self.llm:
            st.error("Componenti LangChain non configurati.")
            return
 
        # Prepara testi per RAG
        all_texts = []
        for filename, doc_data in anonymized_docs.items():
            if doc_data.get('confirmed', False):
                all_texts.append(f"Documento {filename}:\n{doc_data['anonymized']}")
 
        if not all_texts:
            st.warning("Nessun documento confermato per RAG.")
            return
 
        with st.spinner("Creando vector store..."):
            # Chunking
            combined_text = "\n\n".join(all_texts)
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(combined_text)
            
            # Crea FAISS index
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            st.success(f"Vector store con {len(texts)} chunks creato.")
            
            # Setup QA chain
            qa_prompt = """Usa il contesto per rispondere alla domanda.
Se non sai la risposta, dillo chiaramente.

{context}

Domanda: {question}
Risposta:"""
            
            QA_PROMPT = PromptTemplate.from_template(qa_prompt)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
 
    def answer_question(self, query: str) -> str:
        """Risponde usando RAG"""
        if not self.qa_chain:
            return "RAG non pronto. Costruisci prima il knowledge base."
 
        try:
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]
            
            # Aggiungi fonti se disponibili
            source_docs = result.get("source_documents", [])
            if source_docs:
                answer += "\n\n**Fonti:**\n"
                for i, doc in enumerate(source_docs):
                    match = re.search(r"Documento (.*?):\n", doc.page_content)
                    source_info = f" (da {match.group(1)})" if match else ""
                    answer += f"- ...{doc.page_content[-100:]}{source_info}\n"
            
            return answer
        except Exception as e:
            return f"Errore RAG: {e}"
    
    def get_relevant_context(self, query: str, max_docs: int = 3) -> str:
        """Estrae contesto rilevante per query"""
        if not self.vector_store:
            return ""
        
        try:
            docs = self.vector_store.similarity_search(query, k=max_docs)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Errore contesto: {e}"

class CrewAIManager:
    """Manager agenti CrewAI"""
   
    def __init__(self, rag_chatbot: RAGChatbot):
        self.rag_chatbot = rag_chatbot
        self.agents = None
        self.llm = None
        self.setup_crew()
   
    def setup_crew(self):
        """Setup agenti CrewAI"""
        if not Config.AZURE_API_KEY:
            st.warning("Azure non disponibile per CrewAI")
            return
       
        try:
            # LLM per CrewAI
            self.llm = LLM(
                model=f"azure/{Config.DEPLOYMENT_NAME}",
                api_key=Config.AZURE_API_KEY,
                base_url=Config.AZURE_ENDPOINT,
                api_version=Config.AZURE_API_VERSION
            )
           
            # Agenti
            document_analyst = Agent(
                role="Document Analyst",
                goal="Analizzare documenti anonimizzati e fornire insights",
                backstory="Esperto analista documenti con focus su privacy e compliance. "
                         "Lavori solo con documenti anonimizzati per proteggere i dati.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            rag_specialist = Agent(
                role="RAG Specialist",
                goal="Rispondere a domande usando il sistema RAG",
                backstory="Esperto in Information Retrieval e RAG systems. "
                         "Specializzato nel recupero di informazioni da documenti anonimizzati.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            sentiment_analyst = Agent(
                role="Sentiment Analyst",
                goal="Analizzare sentiment e emozioni nei documenti",
                backstory="Esperto in sentiment analysis e behavioral analytics. "
                         "Identifichi emozioni, trend e segnali nei documenti.",
                llm=self.llm,
                verbose=True,
                allow_delegation=False,
                max_iter=3
            )
           
            strategy_coordinator = Agent(
                role="Strategy Coordinator",
                goal="Coordinare analisi e fornire raccomandazioni strategiche",
                backstory="Senior consultant con background in strategic management. "
                         "Traduci insights tecnici in raccomandazioni business concrete.",
                llm=self.llm,
                verbose=True,
                allow_delegation=True,
                max_iter=4
            )
           
            self.agents = {
                'document_analyst': document_analyst,
                'rag_specialist': rag_specialist,
                'sentiment_analyst': sentiment_analyst,
                'strategy_coordinator': strategy_coordinator
            }
           
            st.success("✅ Agenti CrewAI configurati")
           
        except Exception as e:
            st.error(f"Errore setup CrewAI: {e}")
            self.agents = None
   
    def create_analysis_task(self, query: str, analysis_type: str = "comprehensive") -> str:
        """Crea task di analisi per il crew"""
        if not self.agents:
            return "CrewAI non configurato"
       
        try:
            # Ottieni contesto dal RAG
            context = self.rag_chatbot.get_relevant_context(query, max_docs=5)
            
            tasks = []
            
            if analysis_type in ["comprehensive", "document"]:
                # Task analisi documentale
                doc_task = Task(
                    description=f"""
                    Analizza documenti per: {query}
                    
                    CONTESTO: {context}
                    
                    Fornisci:
                    - Tipo e classificazione documenti
                    - Temi e argomenti principali
                    - Elementi rilevanti business
                    - Note compliance
                    """,
                    expected_output="Analisi strutturata con classificazione e insights",
                    agent=self.agents['document_analyst']
                )
                tasks.append(doc_task)
            
            if analysis_type in ["comprehensive", "sentiment"]:
                # Task sentiment
                sentiment_task = Task(
                    description=f"""
                    Analizza sentiment per: {query}
                    
                    CONTESTO: {context}
                    
                    Valuta:
                    - Sentiment generale (scala 1-10)
                    - Emozioni prevalenti
                    - Trend comunicazioni
                    - Segnali rischio/opportunità
                    """,
                    expected_output="Analisi sentiment con valutazioni quantitative",
                    agent=self.agents['sentiment_analyst']
                )
                tasks.append(sentiment_task)
            
            if analysis_type in ["comprehensive", "rag"]:
                # Task RAG
                rag_task = Task(
                    description=f"""
                    Rispondi usando RAG: {query}
                    
                    CONTESTO: {context}
                    
                    Includi:
                    - Risposta diretta
                    - Evidenze documenti
                    - Correlazioni trovate
                    - Informazioni mancanti
                    - Suggerimenti approfondimento
                    """,
                    expected_output="Risposta RAG con evidenze",
                    agent=self.agents['rag_specialist']
                )
                tasks.append(rag_task)
            
            # Task coordinamento (sempre incluso)
            coord_task = Task(
                description=f"""
                Sintetizza risultati per: {query}
                
                Crea sintesi con:
                - Executive Summary (3 punti)
                - Insights strategici 
                - Raccomandazioni prioritarie
                - Next steps concreti
                - Valutazione rischi
                
                Output executive-ready e actionable.
                """,
                expected_output="Sintesi strategica con raccomandazioni",
                agent=self.agents['strategy_coordinator']
            )
            tasks.append(coord_task)
           
            # Crea crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                verbose=True
            )
           
            with st.spinner(f"Eseguendo analisi {analysis_type}..."):
                result = crew.kickoff()
            
            return str(result)
           
        except Exception as e:
            return f"Errore CrewAI: {e}"
    
    def create_custom_task(self, query: str, selected_agents: List[str], custom_instructions: str = "") -> str:
        """Task personalizzate con agenti specifici"""
        if not self.agents:
            return "CrewAI non configurato"
        
        try:
            context = self.rag_chatbot.get_relevant_context(query, max_docs=5)
            
            tasks = []
            agents_to_use = []
            
            for agent_key in selected_agents:
                if agent_key in self.agents:
                    agents_to_use.append(self.agents[agent_key])
                    
                    task = Task(
                        description=f"""
                        {custom_instructions if custom_instructions else f'Analizza secondo il ruolo di {agent_key}'}
                        
                        QUERY: {query}
                        CONTESTO: {context}
                        
                        Fornisci analisi specializzata secondo il tuo ruolo.
                        """,
                        expected_output=f"Analisi specializzata da {agent_key}",
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
            
            with st.spinner(f"Eseguendo task con {len(agents_to_use)} agenti..."):
                result = crew.kickoff()
            
            return str(result)
            
        except Exception as e:
            return f"Errore task personalizzato: {e}"