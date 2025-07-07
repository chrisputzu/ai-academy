"""
Componenti UI riutilizzabili per Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Dict
from config import Config

def setup_page_config():
    """Configura la pagina Streamlit"""
    st.set_page_config(
        page_title="Anonimizzatore Documenti",
        page_icon="🔒",
        layout="wide"
    )

def display_sidebar():
    """Mostra sidebar con configurazioni"""
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Status Azure
        if Config.AZURE_API_KEY and Config.AZURE_ENDPOINT:
            st.success("✅ Azure OpenAI configurato")
            st.info(f"Chat Model: {Config.DEPLOYMENT_NAME}")
            st.info(f"Embedding Model: {Config.AZURE_EMBEDDING_DEPLOYMENT_NAME}")
        else:
            st.error("❌ Azure OpenAI non configurato")
            st.write("Configura le variabili d'ambiente:")
            st.code("""
AZURE_ENDPOINT=your_endpoint
AZURE_API_KEY=your_api_key
AZURE_ENDPOINT_EMB=your_embedding_endpoint
AZURE_API_KEY_EMB=your_embedding_api_key
            """)
        
        st.markdown("---")
        
        # Statistiche documenti
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            st.subheader("📊 Statistiche")
            uploaded_count = len(st.session_state.uploaded_files)
            anonymized_count = len(st.session_state.get('anonymized_docs', {}))
            confirmed_count = sum(1 for doc in st.session_state.get('anonymized_docs', {}).values() 
                                if doc.get('confirmed', False))
            
            st.metric("File caricati", uploaded_count)
            st.metric("Anonimizzati", anonymized_count)
            st.metric("Confermati", confirmed_count)
            
            if confirmed_count > 0:
                if st.session_state.get('vector_store_built', False):
                    st.success("✅ Knowledge Base pronto")
                else:
                    st.info("🔄 Knowledge Base da costruire")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset sessione"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def display_entity_editor(entities: Dict, doc_key: str):
    """Editor per entità rilevate"""
    if not entities:
        st.info("Nessuna entità sensibile rilevata.")
        return entities
    
    st.subheader("🔍 Entità rilevate")
    st.write("Verifica e modifica le entità sensibili:")
    
    current_entities_list = list(entities.items())
    updated_entities_dict = {}
    deleted_placeholders = set()

    for i, (placeholder, original_value) in enumerate(current_entities_list):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.write(f"**{placeholder}**")
        
        with col2:
            new_value = st.text_input(
                "Valore originale",
                value=original_value,
                key=f"{doc_key}_{placeholder}_value_{i}"
            )
            updated_entities_dict[placeholder] = new_value
        
        with col3:
            if st.button("🗑️", key=f"{doc_key}_{placeholder}_delete_{i}", help="Rimuovi"):
                deleted_placeholders.add(placeholder)
    
    # Gestisci cancellazioni
    if deleted_placeholders:
        final_entities = {k: v for k, v in updated_entities_dict.items() 
                         if k not in deleted_placeholders}
        st.session_state.anonymized_docs[doc_key]['entities'] = final_entities
        
        # Re-anonimizza testo
        from anonymizer import NERAnonimizer
        anonymizer = NERAnonimizer()
        st.session_state.anonymized_docs[doc_key]['anonymized'], _ = anonymizer.anonymize(
            st.session_state.anonymized_docs[doc_key]['original']
        )
        st.session_state.vector_store_built = False
        st.rerun()
    
    return updated_entities_dict

def display_file_preview(filename: str, content: str, max_chars: int = 500):
    """Mostra anteprima file"""
    with st.expander(f"📄 {filename} ({len(content)} caratteri)"):
        preview_text = content[:max_chars]
        if len(content) > max_chars:
            preview_text += "..."
        
        st.text_area(
            "Contenuto",
            value=preview_text,
            height=150,
            disabled=True,
            key=f"preview_{filename}",
            label_visibility="collapsed"
        )

def display_analysis_results(filename: str, result: Dict):
    """Mostra risultati analisi"""
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
        
        # Entità
        if result['entities']:
            st.subheader("🔍 Entità Anonimizzate")
            entities_df = pd.DataFrame([
                {
                    'Placeholder': k, 
                    'Valore Originale': v, 
                    'Tipo': k.split('_')[0].replace('[', '')
                }
                for k, v in result['entities'].items()
            ])
            st.dataframe(entities_df, use_container_width=True)

def display_crewai_result(analysis: Dict, index: int):
    """Mostra risultato analisi CrewAI"""
    with st.expander(
        f"🤖 Analisi {index}: {analysis['analysis_type'].upper()} - {analysis['timestamp']}"
    ):
        # Info header
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tipo Analisi", analysis['analysis_type'].capitalize())
        
        with col2:
            st.metric("Timestamp", analysis['timestamp'])
        
        with col3:
            agents_used = analysis.get('agents_used', 'auto')
            if agents_used == 'auto':
                agent_count = "Automatico"
            elif isinstance(agents_used, list):
                agent_count = f"{len(agents_used)} agenti"
            else:
                agent_count = str(agents_used)
            st.metric("Agenti", agent_count)
        
        # Query e risultato
        st.subheader("❓ Query Originale")
        st.info(analysis['query'])
        
        st.subheader("🎯 Risultato Analisi")
        st.markdown(analysis['result'])

def display_progress_metrics():
    """Mostra metriche di progresso"""
    if 'anonymized_docs' in st.session_state:
        confirmed_count = sum(1 for doc in st.session_state.anonymized_docs.values() 
                            if doc.get('confirmed', False))
        total_count = len(st.session_state.anonymized_docs)
        
        if total_count > 0:
            st.metric(
                "Progresso Conferme",
                f"{confirmed_count}/{total_count}",
                delta=f"{(confirmed_count/total_count)*100:.1f}%"
            )

def display_examples_section():
    """Mostra esempi di query CrewAI"""
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

def create_download_button(content: str, filename: str, button_text: str, key: str):
    """Crea un bottone per il download di contenuti"""
    try:
        # Usa direttamente st.download_button invece di gestire file temporanei
        st.download_button(
            label=button_text,
            data=content,
            file_name=filename,
            mime="application/json",
            key=key
        )
    except Exception as e:
        st.error(f"Errore creazione bottone download: {e}")
        st.warning("Riprova più tardi o contatta il supporto.")

def display_semantic_analysis_results(evaluation):
    """Visualizza i risultati dell'analisi semantica"""
    
    # Punteggio bias
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Bias Score",
            f"{evaluation.bias_score:.2f}",
            delta=f"{-evaluation.semantic_improvement:.2f}" if evaluation.semantic_improvement > 0 else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Rigenerazioni",
            evaluation.regeneration_count
        )
    
    with col3:
        status_color = {
            "approved": "🟢",
            "log": "🟡",
            "warn": "🟠",
            "regenerate": "🔴"
        }.get(evaluation.action_taken, "⚪")
        
        st.metric(
            "Stato",
            f"{status_color} {evaluation.action_taken.title()}"
        )
    
    # Dettagli analisi
    with st.expander("🔍 Analisi Semantica Dettagliata", expanded=True):
        st.write("**Comprensione Contestuale:**")
        st.write(evaluation.contextual_understanding)
        
        st.write("**Assunzioni Implicite:**")
        for assumption in evaluation.implicit_assumptions:
            st.write(f"• {assumption}")
        
        st.write("**Dinamiche di Potere:**")
        for key, value in evaluation.power_dynamics.items():
            st.write(f"• {key}: {value}")
        
        st.write("**Analisi Rappresentazionale:**")
        for key, value in evaluation.representation_analysis.items():
            st.write(f"• {key}: {value}")
        
        st.write("**Implicazioni Linguistiche:**")
        for implication in evaluation.language_implications:
            st.write(f"• {implication}")

def display_semantic_stats(stats: Dict):
    """Visualizza statistiche dell'analisi semantica"""
    if not stats:
        st.info("Nessuna analisi semantica effettuata")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Totale Analisi",
            stats["total_evaluations"]
        )
        st.metric(
            "Bias Rilevati",
            stats["responses_with_bias"],
            f"{stats['bias_percentage']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Score Medio",
            f"{stats['avg_bias_score']:.2f}"
        )
        st.metric(
            "Rigenerazioni",
            stats["responses_regenerated"]
        )
    
    with col3:
        st.metric(
            "Miglioramento Medio",
            f"{stats['avg_semantic_improvement']:.2%}"
        )
        st.metric(
            "Tasso Successo",
            f"{stats['semantic_success_rate']:.1f}%"
        )

def display_semantic_examples():
    """Mostra esempi per test analisi semantica"""
    with st.expander("💡 Esempi per Test", expanded=False):
        st.write("**Esempi di bias semantici:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🔴 **Bias Semantici Sottili:**")
            st.write("• 'È naturale che gli uomini siano più competitivi'")
            st.write("• 'Le donne tendono a essere più empatiche'")
            st.write("• 'I giovani ovviamente sono più tecnologici'")
            
            if st.button("📋 Usa Esempio Bias", key="bias_example"):
                st.session_state.semantic_query = "Perché le donne sono meno presenti in ruoli di leadership?"
                st.session_state.semantic_response = "È naturale che ci siano meno donne in posizioni di leadership perché le donne tendono a essere più orientate alla famiglia e meno competitive. Gli uomini sono biologicamente più portati alla competizione e al dominio, quindi è logico che occupino più ruoli dirigenziali."
        
        with col2:
            st.write("🟢 **Risposte Equilibrate:**")
            st.write("• 'La presenza varia per diversi fattori'")
            st.write("• 'Le differenze individuali sono significative'")
            st.write("• 'Molteplici fattori influenzano le preferenze'")
            
            if st.button("📋 Usa Esempio Neutro", key="neutral_example"):
                st.session_state.semantic_query = "Come si distribuiscono i ruoli di leadership?"
                st.session_state.semantic_response = "La distribuzione dei ruoli di leadership è influenzata da molteplici fattori sociali, economici e organizzativi. Le barriere strutturali, i bias inconsci e le pratiche organizzative possono impattare le opportunità di carriera. Un approccio inclusivo e basato sulle competenze può favorire una leadership più diversificata."