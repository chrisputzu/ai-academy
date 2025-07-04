"""
Funzioni utility e gestione stato sessione.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from anonymizer import NERAnonimizer
from ai_processor import AzureProcessor, RAGChatbot, CrewAIManager

def init_session_state():
    """Inizializza stato sessione"""
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

def validate_file_upload(uploaded_file) -> bool:
    """Valida file caricato"""
    if not uploaded_file:
        return False
    
    # Controlla estensione
    if not uploaded_file.name.endswith('.txt'):
        st.error("Solo file .txt sono supportati")
        return False
    
    # Controlla dimensione (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File troppo grande (max 10MB)")
        return False
    
    return True

def process_uploaded_files(uploaded_files):
    """Processa file caricati"""
    new_files_uploaded = False
    
    for file in uploaded_files:
        if validate_file_upload(file) and file.name not in st.session_state.uploaded_files:
            try:
                content = file.read().decode('utf-8')
                st.session_state.uploaded_files[file.name] = {
                    'content': content,
                    'size': len(content)
                }
                new_files_uploaded = True
            except Exception as e:
                st.error(f"Errore lettura file {file.name}: {e}")
    
    if new_files_uploaded:
        # Reset stato quando si caricano nuovi file
        st.session_state.anonymized_docs = {}
        st.session_state.processed_docs = {}
        st.session_state.vector_store_built = False
        st.session_state.chat_history = []
        st.session_state.crewai_history = []
        return True
    
    return False

def run_anonymization():
    """Esegue anonimizzazione su tutti i file"""
    if not st.session_state.uploaded_files:
        st.warning("Nessun file caricato")
        return
    
    progress_bar = st.progress(0)
    total_files = len(st.session_state.uploaded_files)
    
    for i, (filename, file_data) in enumerate(st.session_state.uploaded_files.items()):
        progress_bar.progress((i + 1) / total_files, f"Processando {filename}...")
        
        # Anonimizza
        anonymized_text, entities = st.session_state.anonymizer.anonymize(file_data['content'])
        
        st.session_state.anonymized_docs[filename] = {
            'original': file_data['content'],
            'anonymized': anonymized_text,
            'entities': entities,
            'confirmed': False
        }
    
    progress_bar.empty()
    st.success("✅ Anonimizzazione completata!")
    st.session_state.vector_store_built = False

def run_ai_analysis():
    """Esegue analisi AI sui documenti confermati"""
    confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() 
                     if v.get('confirmed', False)}
    
    if not confirmed_docs:
        st.warning("Nessun documento confermato")
        return
    
    progress_bar = st.progress(0)
    
    for i, (filename, doc_data) in enumerate(confirmed_docs.items()):
        progress_bar.progress((i + 1) / len(confirmed_docs), f"Analizzando {filename}...")
        
        # Analisi Azure
        analysis = st.session_state.processor.process_document(doc_data['anonymized'])
        
        st.session_state.processed_docs[filename] = {
            'anonymized_text': doc_data['anonymized'],
            'entities_count': len(doc_data['entities']),
            'analysis': analysis,
            'entities': doc_data['entities']
        }
    
    progress_bar.empty()
    st.success("✅ Analisi completata!")

def build_rag_knowledge_base():
    """Costruisce knowledge base RAG"""
    confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() 
                     if v.get('confirmed', False)}
    
    if not confirmed_docs:
        st.warning("Nessun documento confermato per RAG")
        return False
    
    if not st.session_state.vector_store_built:
        with st.spinner("Costruendo knowledge base..."):
            st.session_state.rag_chatbot.build_vector_store(confirmed_docs)
            st.session_state.vector_store_built = True
            return True
    
    return True

def export_results_json(results: dict, filename_prefix: str) -> str:
    """Esporta risultati in JSON"""
    export_data = {
        **results,
        'metadata': {
            'exported_at': datetime.now().isoformat(),
            'total_items': len(results) if isinstance(results, dict) else 1
        }
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)

def get_confirmed_docs_count() -> int:
    """Ritorna numero documenti confermati"""
    if 'anonymized_docs' not in st.session_state:
        return 0
    
    return sum(1 for doc in st.session_state.anonymized_docs.values() 
              if doc.get('confirmed', False))

def reset_document_state(filename: str):
    """Reset stato documento specifico"""
    if filename in st.session_state.uploaded_files:
        original_data = st.session_state.uploaded_files[filename]
        anonymized_text, entities = st.session_state.anonymizer.anonymize(original_data['content'])
        
        st.session_state.anonymized_docs[filename] = {
            'original': original_data['content'],
            'anonymized': anonymized_text,
            'entities': entities,
            'confirmed': False
        }
        st.session_state.vector_store_built = False

def add_chat_message(role: str, content: str):
    """Aggiunge messaggio alla chat history"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })

def add_crewai_result(query: str, analysis_type: str, result: str, agents_used=None):
    """Aggiunge risultato CrewAI alla history"""
    analysis_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "analysis_type": analysis_type,
        "result": result,
        "agents_used": agents_used if agents_used else "auto"
    }
    
    st.session_state.crewai_history.append(analysis_result)

def clear_chat_history():
    """Pulisce cronologia chat"""
    st.session_state.chat_history = []

def clear_crewai_history():
    """Pulisce cronologia CrewAI"""
    st.session_state.crewai_history = []

def get_system_stats() -> dict:
    """Ritorna statistiche sistema"""
    return {
        'uploaded_files': len(st.session_state.get('uploaded_files', {})),
        'anonymized_docs': len(st.session_state.get('anonymized_docs', {})),
        'confirmed_docs': get_confirmed_docs_count(),
        'processed_docs': len(st.session_state.get('processed_docs', {})),
        'chat_messages': len(st.session_state.get('chat_history', [])),
        'crewai_analyses': len(st.session_state.get('crewai_history', [])),
        'vector_store_ready': st.session_state.get('vector_store_built', False)
    }