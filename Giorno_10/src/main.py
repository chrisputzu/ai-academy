"""
App principale Streamlit per l'anonimizzazione documenti.
"""

import streamlit as st
import json
import pandas as pd
from ui_components import (
    setup_page_config, display_sidebar, display_entity_editor,
    display_file_preview, display_analysis_results, display_crewai_result,
    display_progress_metrics, display_examples_section, create_download_button
)
from utils import (
    init_session_state, process_uploaded_files, run_anonymization,
    run_ai_analysis, build_rag_knowledge_base, export_results_json,
    get_confirmed_docs_count, reset_document_state, add_chat_message,
    add_crewai_result, clear_crewai_history
)

def main():
    """Funzione principale dell'app"""
    
    # Setup
    setup_page_config()
    init_session_state()
    
    # Header
    st.title("🔒 Anonimizzatore Documenti con NER, RAG e CrewAI")
    st.markdown("---")
    
    # Sidebar
    display_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload", 
        "🔍 Anonimizzazione", 
        "📊 Analisi", 
        "💬 Chatbot RAG", 
        "🤖 CrewAI"
    ])
    
    # TAB 1: Upload
    with tab1:
        upload_tab()
    
    # TAB 2: Anonimizzazione
    with tab2:
        anonymization_tab()
    
    # TAB 3: Analisi
    with tab3:
        analysis_tab()
    
    # TAB 4: RAG
    with tab4:
        rag_tab()
    
    # TAB 5: CrewAI
    with tab5:
        crewai_tab()

def upload_tab():
    """Tab per upload file"""
    st.header("📤 Carica Documenti")
    
    uploaded_files = st.file_uploader(
        "Carica uno o più file .txt",
        type=['txt'],
        accept_multiple_files=True,
        help="Seleziona i file di testo da anonimizzare"
    )
    
    if uploaded_files:
        if process_uploaded_files(uploaded_files):
            st.success(f"Caricati {len(uploaded_files)} file")
            st.rerun()
        else:
            st.info("Nessun nuovo file caricato.")
        
        # Mostra anteprima
        st.subheader("📄 File caricati")
        for filename, file_data in st.session_state.uploaded_files.items():
            display_file_preview(filename, file_data['content'])

def anonymization_tab():
    """Tab per anonimizzazione"""
    st.header("🔍 Anonimizzazione e Revisione")
    
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Carica prima alcuni documenti nella tab 'Upload'")
        return
    
    # Bottone anonimizzazione
    if st.button("🚀 Avvia Anonimizzazione", type="primary"):
        run_anonymization()
        st.rerun()
    
    # Mostra documenti anonimizzati
    if st.session_state.anonymized_docs:
        st.subheader("📝 Revisiona Documenti Anonimizzati")
        
        for filename, doc_data in st.session_state.anonymized_docs.items():
            with st.expander(
                f"📄 {filename} {'✅' if doc_data['confirmed'] else '⏳'}", 
                expanded=not doc_data['confirmed']
            ):
                
                col1, col2 = st.columns(2)
                
                # Testo originale
                with col1:
                    st.write("**Testo Originale:**")
                    preview = doc_data['original'][:300]
                    if len(doc_data['original']) > 300:
                        preview += "..."
                    
                    st.text_area(
                        "Originale",
                        value=preview,
                        height=200,
                        disabled=True,
                        key=f"orig_{filename}",
                        label_visibility="collapsed"
                    )
                
                # Testo anonimizzato
                with col2:
                    st.write("**Testo Anonimizzato:**")
                    edited_text = st.text_area(
                        "Anonimizzato (modificabile)",
                        value=doc_data['anonymized'],
                        height=200,
                        key=f"anon_{filename}",
                        label_visibility="collapsed"
                    )
                    
                    # Aggiorna se modificato
                    if edited_text != doc_data['anonymized']:
                        st.session_state.anonymized_docs[filename]['anonymized'] = edited_text
                
                # Editor entità
                updated_entities = display_entity_editor(dict(doc_data['entities']), filename)
                
                # Bottoni azione
                col_confirm, col_reset = st.columns(2)
                
                with col_confirm:
                    if st.button(f"✅ Conferma {filename}", key=f"confirm_{filename}"):
                        st.session_state.anonymized_docs[filename]['confirmed'] = True
                        st.session_state.anonymized_docs[filename]['entities'] = updated_entities
                        st.success(f"✅ {filename} confermato!")
                        st.session_state.vector_store_built = False
                        st.rerun()
                
                with col_reset:
                    if st.button(f"🔄 Reset {filename}", key=f"reset_{filename}"):
                        reset_document_state(filename)
                        st.rerun()
        
        # Statistiche progresso
        display_progress_metrics()

def analysis_tab():
    """Tab per analisi AI"""
    st.header("📊 Analisi AI")
    
    confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() 
                     if v.get('confirmed', False)}
    
    if not confirmed_docs:
        st.warning("⚠️ Conferma prima alcuni documenti anonimizzati")
        return
    
    st.write(f"Documenti confermati pronti: **{len(confirmed_docs)}**")
    
    if st.button("🤖 Avvia Analisi AI", type="primary"):
        run_ai_analysis()
    
    # Mostra risultati
    if st.session_state.processed_docs:
        st.subheader("📋 Risultati Analisi")
        
        for filename, result in st.session_state.processed_docs.items():
            display_analysis_results(filename, result)
            
            # Download JSON
            result_json = export_results_json({
                'filename': filename,
                'anonymized_text': result['anonymized_text'],
                'analysis': result['analysis'],
                'entities': result['entities'],
                'entities_count': result['entities_count']
            }, f"analisi_{filename}")
            
            create_download_button(
                result_json,
                f"analisi_{filename}.json",
                f"💾 Scarica {filename}",
                f"download_{filename}"
            )

def rag_tab():
    """Tab per RAG chatbot"""
    st.header("💬 Chatta con i Documenti")
    
    confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() 
                     if v.get('confirmed', False)}
    
    if not confirmed_docs:
        st.warning("⚠️ Carica e conferma documenti per abilitare il chatbot")
        return
    
    # Costruisci knowledge base
    if build_rag_knowledge_base():
        st.info(f"Chatbot pronto per {len(confirmed_docs)} documenti")
        
        # Mostra cronologia chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input utente
        if prompt := st.chat_input("Fai una domanda sui documenti..."):
            # Aggiungi messaggio utente
            add_chat_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Genera risposta
            with st.chat_message("assistant"):
                with st.spinner("Generando risposta..."):
                    response = st.session_state.rag_chatbot.answer_question(prompt)
                    st.markdown(response)
            
            # Aggiungi risposta
            add_chat_message("assistant", response)
    else:
        st.error("Impossibile costruire knowledge base. Verifica configurazione Azure.")

def crewai_tab():
    """Tab per CrewAI"""
    st.header("🤖 Analisi Multi-Agente CrewAI")
    
    confirmed_docs = {k: v for k, v in st.session_state.anonymized_docs.items() 
                     if v.get('confirmed', False)}
    
    if not confirmed_docs:
        st.warning("⚠️ Conferma documenti per abilitare CrewAI")
        return
    
    if not st.session_state.crewai_manager.agents:
        st.error("❌ CrewAI non configurato. Verifica Azure OpenAI.")
        return
    
    # Assicura knowledge base
    build_rag_knowledge_base()
    
    st.success(f"🎯 CrewAI pronto per {len(confirmed_docs)} documenti")
    
    # Configurazione analisi
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
            }[x]
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
    
    # Query input
    st.subheader("❓ Query per l'Analisi")
    query_input = st.text_area(
        "Inserisci la tua domanda:",
        placeholder="Es: Analizza i temi principali e identifica rischi operativi...",
        height=100
    )
    
    # Istruzioni personalizzate
    if analysis_type == "custom":
        custom_instructions = st.text_area(
            "Istruzioni Personalizzate:",
            placeholder="Istruzioni specifiche per gli agenti...",
            height=80
        )
    else:
        custom_instructions = ""
    
    # Bottoni
    col_analyze, col_clear = st.columns(2)
    
    with col_analyze:
        if st.button("🚀 Avvia Analisi CrewAI", type="primary", disabled=not query_input.strip()):
            if analysis_type == "custom" and not selected_agents:
                st.error("Seleziona almeno un agente")
            else:
                # Esegui analisi
                if analysis_type == "custom":
                    result = st.session_state.crewai_manager.create_custom_task(
                        query_input, selected_agents, custom_instructions
                    )
                else:
                    result = st.session_state.crewai_manager.create_analysis_task(
                        query_input, analysis_type
                    )
                
                # Salva risultato
                add_crewai_result(query_input, analysis_type, result, selected_agents)
                st.success("✅ Analisi CrewAI completata!")
    
    with col_clear:
        if st.button("🗑️ Pulisci Cronologia"):
            clear_crewai_history()
            st.success("Cronologia pulita!")
            st.rerun()
    
    # Mostra risultati
    if st.session_state.crewai_history:
        st.subheader("📋 Risultati Analisi CrewAI")
        
        for i, analysis in enumerate(reversed(st.session_state.crewai_history)):
            display_crewai_result(analysis, len(st.session_state.crewai_history) - i)
            
            # Download
            result_json = export_results_json(analysis, f"crewai_analysis_{i}")
            create_download_button(
                result_json,
                f"crewai_analysis_{analysis['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                "💾 Scarica Risultato",
                f"download_crewai_{i}"
            )
    
    # Esempi
    display_examples_section()

if __name__ == "__main__":
    main()