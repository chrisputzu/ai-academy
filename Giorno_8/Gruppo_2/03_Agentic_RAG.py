#Progetto Agentic RAG 
#

import os
import re
import json
from typing import Dict, List, Tuple
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool, RagTool
from crewai.llm import LLM

from transformers import pipeline

from openai import AzureOpenAI

from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configurazione centralizzata"""
    NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
    DOCUMENTS_DIR = "04_documenti"
    
    # Azure OpenAI Configuration (se disponibile)
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION = "2024-12-01-preview"
    DEPLOYMENT_NAME = "gpt-4.1"
    

# Forza OPENAI_API_KEY nell'ambiente per compatibilità con openai lib
os.environ["OPENAI_API_KEY"] = Config.AZURE_API_KEY


class NERAnonimizer:
    """Classe per anonimizzazione con NER e regex"""
    
    def __init__(self):
        self.ner_pipe = pipeline(
            "ner", 
            model=Config.NER_MODEL, 
            aggregation_strategy="simple"
        )
        
        self.regex_patterns = {
            "IBAN": r'\bIT\d{2}[A-Z0-9]{23}\b',
            "EMAIL": r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
            "CF": r'\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b',
            "CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
    
    def mask_with_regex(self, text: str) -> str:
        """Applica mascheramento con regex"""
        for label, pattern in self.regex_patterns.items():
            text = re.sub(pattern, f"[{label}]", text, flags=re.IGNORECASE)
        return text
    
    def mask_with_ner(self, text: str) -> Tuple[str, Dict]:
        """Applica mascheramento con NER"""
        try:
            entities = self.ner_pipe(text)
            entity_map = {}
            
            # Ordina entità per posizione (dal fondo per evitare spostamenti di indici)
            sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
            
            for ent in sorted_entities:
                label = ent['entity_group']
                original_text = ent['word']
                
                # Salva mapping per eventuale ricostruzione
                entity_map[f"[{label}_{len(entity_map)}]"] = original_text
                
                # Sostituisci nel testo
                placeholder = f"[{label}_{len(entity_map)-1}]"
                text = text[:ent['start']] + placeholder + text[ent['end']:]
            
            return text, entity_map
            
        except Exception as e:
            print(f"Errore NER: {e}")
            return text, {}
    
    def anonymize(self, text: str) -> Tuple[str, Dict]:
        """Pipeline completa di anonimizzazione"""
        # Step 1: Regex patterns
        masked_text = self.mask_with_regex(text)
        
        # Step 2: NER
        final_text, entity_map = self.mask_with_ner(masked_text)
        
        return final_text, entity_map

class AzureProcessor:
    """Processore Azure OpenAI con fallback"""
    
    def __init__(self):
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup client con fallback"""
        if Config.AZURE_API_KEY and Config.AZURE_ENDPOINT:
            try:
                self.client = AzureOpenAI(
                    api_key=Config.AZURE_API_KEY,
                    api_version=Config.AZURE_API_VERSION,
                    azure_endpoint=Config.AZURE_ENDPOINT
                )
                print("Azure OpenAI configurato")
            except Exception as e:
                print(f"Errore Azure OpenAI: {e}")
                self.client = None
        
    
    def process_document(self, anonymized_text: str) -> str:
        """Processa documento con AI o fallback locale"""
        
        try:
            messages = [
    {
        "role": "system",
        "content": "Sei un assistente intelligente che riceve documenti anonimizzati e svolge le seguenti operazioni:\n\n"
                   "1. Fornisci il tipo di documento (es: email, fattura, contratto, altro).\n" 
                   "2. Fornisci un riepilogo chiaro e conciso del contenuto (max 5 righe).\n"
                   "3. Esegui un’analisi semantica del testo, evidenziando i temi principali, sentimenti, e intenzioni implicite.\n"
                   "4. Se il documento è una comunicazione da o verso un cliente, genera una possibile risposta adeguata, professionale e contestuale.\n"
                   "5. Lavora solo con i contenuti presenti nel documento anonimizzato, evitando ogni supposizione che possa violare la privacy.\n\n"
                   "Assicurati che tutte le risposte siano compatibili con contesti dove i dati personali sono stati rimossi per protezione della privacy."
    },
    {
        "role": "user",
        "content": f"Analizza questo documento:\n\n{anonymized_text}"
    }
]
            
            if isinstance(self.client, AzureOpenAI):
                response = self.client.chat.completions.create(
                    model=Config.DEPLOYMENT_NAME,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
        
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Errore AI: {e}")
    

class DocumentProcessor(RagTool):
    """Tool personalizzato per CrewAI"""

    name: str = "document_processor"
    description: str = "Processa documenti con anonimizzazione NER e analisi Azure"

    def _run(self, file_path: str) -> str:
        """Esegue la pipeline completa su un file"""
        try:
            # Inizializza qui per evitare errori di istanziazione multipla
            anonymizer = NERAnonimizer()
            azure_processor = AzureProcessor()

            # Leggi file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Anonimizza
            anon_text, entity_map = anonymizer.anonymize(content)

            # Processa con Azure
            azure_result = azure_processor.process_document(anon_text)

            # Prepara risultato
            result = {
                "file": file_path,
                "original_length": len(content),
                "anonymized_length": len(anon_text),
                "entities_found": len(entity_map),
                "analysis": azure_result,
                "entity_map": entity_map
            }

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"Errore nel processamento di {file_path}: {e}"

def setup_crew():
    """Configura e restituisce il crew"""
    
    # Tools
    dir_tool = DirectoryReadTool(directory=Config.DOCUMENTS_DIR)
    file_tool = FileReadTool()
    doc_processor = DocumentProcessor()
    
    # LLM configuration per CrewAI
    llm = None

    try:
        if Config.AZURE_API_KEY:
            llm = LLM(
                model="azure/" + Config.DEPLOYMENT_NAME,
                api_key=Config.AZURE_API_KEY,
                base_url=Config.AZURE_ENDPOINT,
                api_version=Config.AZURE_API_VERSION
            )
        
    except Exception as e:
        print(f"Errore configurazione LLM: {e}")
        llm = None
    
    # Agent (funziona anche senza LLM)
    document_analyst = Agent(
        role="Document Privacy Analyst",
        goal="Processare documenti garantendo privacy e fornendo analisi utili",
        backstory="""Sei un esperto analista di documenti specializzato in 
        protezione della privacy. Utilizzi tecniche avanzate di NER per 
        anonimizzare i dati sensibili prima dell'analisi.""",
        tools=[dir_tool, file_tool, doc_processor],
        llm=llm,  # Può essere None
        verbose=True,
        allow_delegation=False
    )
    
    # Task
    analysis_task = Task(
        description="""
        1. Esplora la directory dei documenti
        2. Per ogni documento trovato:
           - Applica anonimizzazione con NER
           - Processa con AI o analisi locale
           - Genera report completo
        3. Fornisci un riassunto finale di tutti i documenti processati
        4. Fornisci una risposta contestuale se il documento è una comunicazione con un cliente
        """,
        expected_output="Report dettagliato dell'analisi di tutti i documenti con protezione della privacy e risposte contestuali",
        agent=document_analyst
    )
    
    # Crew
    crew = Crew(
        agents=[document_analyst],
        tasks=[analysis_task],
        verbose=True
    )
    
    return crew

def main():
    """Funzione principale"""
    print("Avvio Pipeline Collaborativa NER + CrewAI")
    
    # Verifica e crea directory se necessario
    docs_path = Path(Config.DOCUMENTS_DIR)
    if not docs_path.exists():
        print(f"Creando directory {Config.DOCUMENTS_DIR}")
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Crea file di esempio se la directory è vuota
        sample_files = {
            "Mail.txt": "Gentile cliente Mario Rossi, la sua richiesta è stata elaborata. IBAN: IT60X0542811101000000123456",
            "nota_di_credito.txt": "Nota di credito per il cliente con CF: RSSMRA80A01H501Z, importo €150,00",  
            "Fattura.txt": "Fattura n.123 per mario.rossi@email.com, carta: 1234-5678-9012-3456"
        }
        
        for filename, content in sample_files.items():
            (docs_path / filename).write_text(content, encoding='utf-8')
        
        print(f"Creati {len(sample_files)} file di esempio")
    
    # Verifica presenza documenti (cerca tutti i file di testo)
    doc_files = list(docs_path.glob("*.txt"))
    if not doc_files:
        print(f"Nessun documento .txt trovato in {Config.DOCUMENTS_DIR}")
        print("Aggiungi alcuni file .txt per testare la pipeline")
        return
    
    print(f"Trovati {len(doc_files)} documenti da processare:")
    for file in doc_files:
        print(f"   - {file.name}")
    
    try:
        # Setup e avvio crew
        crew = setup_crew()
        result = crew.kickoff()
        
        print("\n" + "="*50)
        print("RISULTATI FINALI")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        print(f"Suggerimento: Verifica le credenziali Azure nella classe Config")

if __name__ == "__main__":
    main()