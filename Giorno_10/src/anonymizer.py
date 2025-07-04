"""
Sistema di anonimizzazione con NER e regex.
"""

import re
from typing import Dict, Tuple
from transformers import pipeline
import streamlit as st
from config import Config, REGEX_PATTERNS

class NERAnonimizer:
    """Anonimizzatore con NER e regex"""
    
    def __init__(self):
        self.regex_patterns = REGEX_PATTERNS
        self._ner_pipe = None
    
    @property
    def ner_pipe(self):
        """Lazy loading del modello NER"""
        if self._ner_pipe is None:
            with st.spinner("Caricamento modello NER..."):
                try:
                    self._ner_pipe = pipeline(
                        "ner",
                        model=Config.NER_MODEL,
                        aggregation_strategy="simple"
                    )
                except Exception as e:
                    st.error(f"Errore caricamento NER: {e}")
                    return None
        return self._ner_pipe
    
    def mask_with_regex(self, text: str) -> Tuple[str, Dict]:
        """Applica mascheramento con regex"""
        masked_text = text
        found_entities = {}
        
        # Ordina pattern per lunghezza (più lunghi prima)
        sorted_patterns = sorted(
            self.regex_patterns.items(), 
            key=lambda item: len(item[1]), 
            reverse=True
        )

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
        if not self.ner_pipe:
            return text, {}
            
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
        if not text or not text.strip():
            return text, {}
        
        # Regex prima, poi NER
        masked_text, regex_entities = self.mask_with_regex(text)
        final_text, ner_entities = self.mask_with_ner(masked_text)
        
        # Combina entità
        all_entities = {**regex_entities, **ner_entities}
        
        return final_text, all_entities