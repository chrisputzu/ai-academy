"""
Test per sistema anonimizzazione.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest.mock import Mock, patch
from anonymizer import NERAnonimizer

class TestNERAnonimizer:
    """Test classe NERAnonimizer"""
    
    def test_init(self):
        """Test inizializzazione"""
        anonymizer = NERAnonimizer()
        assert anonymizer.regex_patterns is not None
        assert anonymizer._ner_pipe is None
    
    @patch('anonymizer.pipeline')
    def test_ner_pipe_lazy_loading(self, mock_pipeline, mock_streamlit):
        """Test lazy loading del modello NER"""
        anonymizer = NERAnonimizer()
        
        # Prima chiamata - dovrebbe caricare il modello
        pipe = anonymizer.ner_pipe
        assert mock_pipeline.called
        
        # Seconda chiamata - dovrebbe usare cache
        mock_pipeline.reset_mock()
        pipe2 = anonymizer.ner_pipe
        assert not mock_pipeline.called
        assert pipe == pipe2
    
    def test_mask_with_regex_basic(self, sample_text):
        """Test mascheramento regex base"""
        anonymizer = NERAnonimizer()
        
        masked_text, entities = anonymizer.mask_with_regex(sample_text)
        
        # Verifica che abbia trovato entità
        assert len(entities) > 0
        
        # Verifica che le entità siano nel formato corretto
        for placeholder, original in entities.items():
            assert placeholder.startswith('[')
            assert placeholder.endswith(']')
            assert '_' in placeholder
            assert original in sample_text
            assert placeholder in masked_text
    
    def test_mask_with_regex_iban(self):
        """Test mascheramento IBAN specifico"""
        anonymizer = NERAnonimizer()
        text = "Il mio IBAN è IT60 X054 2811 1010 0000 0123 456 per i pagamenti"
        
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        # Dovrebbe trovare l'IBAN
        iban_entities = [k for k in entities.keys() if k.startswith('[IBAN_')]
        assert len(iban_entities) == 1
        
        iban_placeholder = iban_entities[0]
        assert entities[iban_placeholder] == "IT60 X054 2811 1010 0000 0123 456"
        assert iban_placeholder in masked_text
    
    def test_mask_with_regex_email(self):
        """Test mascheramento email"""
        anonymizer = NERAnonimizer()
        text = "Contattami su mario.rossi@example.com o test@domain.co.uk"
        
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        # Dovrebbe trovare 2 email
        email_entities = [k for k in entities.keys() if k.startswith('[EMAIL_')]
        assert len(email_entities) == 2
        
        email_values = [entities[k] for k in email_entities]
        assert "mario.rossi@example.com" in email_values
        assert "test@domain.co.uk" in email_values
    
    def test_mask_with_regex_cf(self):
        """Test mascheramento codice fiscale"""
        anonymizer = NERAnonimizer()
        text = "Il codice fiscale è RSSMRA80A01H501Z"
        
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        cf_entities = [k for k in entities.keys() if k.startswith('[CF_')]
        assert len(cf_entities) == 1
        assert entities[cf_entities[0]] == "RSSMRA80A01H501Z"
    
    def test_mask_with_regex_empty_text(self, sample_empty_text):
        """Test con testo vuoto"""
        anonymizer = NERAnonimizer()
        
        masked_text, entities = anonymizer.mask_with_regex(sample_empty_text)
        
        assert masked_text == sample_empty_text
        assert len(entities) == 0
    
    def test_mask_with_regex_no_entities(self, sample_text_no_entities):
        """Test con testo senza entità"""
        anonymizer = NERAnonimizer()
        
        masked_text, entities = anonymizer.mask_with_regex(sample_text_no_entities)
        
        assert masked_text == sample_text_no_entities
        assert len(entities) == 0
    
    def test_mask_with_ner_success(self, mock_ner_pipeline, mock_streamlit):
        """Test mascheramento NER con successo"""
        anonymizer = NERAnonimizer()
        anonymizer._ner_pipe = mock_ner_pipeline
        
        text = "Mario Rossi lavora in ACME SpA"
        
        masked_text, entities = anonymizer.mask_with_ner(text)
        
        # Verifica chiamata al modello
        assert mock_ner_pipeline.called
        
        # Verifica entità trovate
        assert len(entities) == 2
        per_entities = [k for k in entities.keys() if k.startswith('[PER_')]
        org_entities = [k for k in entities.keys() if k.startswith('[ORG_')]
        
        assert len(per_entities) == 1
        assert len(org_entities) == 1
    
    def test_mask_with_ner_no_model(self, mock_streamlit):
        """Test NER senza modello caricato"""
        anonymizer = NERAnonimizer()
        anonymizer._ner_pipe = None
        
        text = "Mario Rossi lavora in ACME SpA"
        
        masked_text, entities = anonymizer.mask_with_ner(text)
        
        # Dovrebbe ritornare testo invariato
        assert masked_text == text
        assert len(entities) == 0
    
    def test_mask_with_ner_low_confidence(self, mock_streamlit):
        """Test NER con confidence bassa"""
        anonymizer = NERAnonimizer()
        
        # Mock con score basso
        mock_pipe = Mock()
        mock_pipe.return_value = [
            {
                'entity_group': 'PER',
                'score': 0.3,  # Sotto threshold (0.5)
                'start': 0,
                'end': 11,
                'word': 'Mario Rossi'
            }
        ]
        anonymizer._ner_pipe = mock_pipe
        
        text = "Mario Rossi"
        masked_text, entities = anonymizer.mask_with_ner(text)
        
        # Non dovrebbe mascherare con confidence bassa
        assert masked_text == text
        assert len(entities) == 0
    
    def test_anonymize_complete_pipeline(self, sample_text, mock_ner_pipeline, mock_streamlit):
        """Test pipeline completa di anonimizzazione"""
        anonymizer = NERAnonimizer()
        anonymizer._ner_pipe = mock_ner_pipeline
        
        anonymized_text, all_entities = anonymizer.anonymize(sample_text)
        
        # Verifica che sia diverso dall'originale
        assert anonymized_text != sample_text
        
        # Verifica che contenga placeholder
        assert '[' in anonymized_text and ']' in anonymized_text
        
        # Verifica che abbia trovato entità da entrambi i sistemi
        assert len(all_entities) > 0
        
        # Verifica mix di entità regex e NER
        regex_entities = [k for k in all_entities.keys() 
                         if any(k.startswith(f'[{t}_') for t in ['IBAN', 'EMAIL', 'CF', 'CARD', 'PHONE'])]
        ner_entities = [k for k in all_entities.keys() 
                       if any(k.startswith(f'[{t}_') for t in ['PER', 'ORG'])]
        
        assert len(regex_entities) > 0  # Dovrebbe trovare entità regex
        assert len(ner_entities) > 0    # Dovrebbe trovare entità NER
    
    def test_anonymize_empty_text(self, sample_empty_text):
        """Test anonimizzazione testo vuoto"""
        anonymizer = NERAnonimizer()
        
        anonymized_text, entities = anonymizer.anonymize(sample_empty_text)
        
        assert anonymized_text == sample_empty_text
        assert len(entities) == 0
    
    def test_anonymize_preserves_structure(self, mock_streamlit):
        """Test che l'anonimizzazione preservi la struttura del testo"""
        anonymizer = NERAnonimizer()
        
        text = """Documento importante
        
        Dati cliente:
        - Nome: Mario Rossi
        - Email: mario@test.com
        
        Fine documento."""
        
        anonymized_text, entities = anonymizer.anonymize(text)
        
        # Dovrebbe preservare newline e struttura
        assert '\n' in anonymized_text
        assert 'Documento importante' in anonymized_text
        assert 'Fine documento.' in anonymized_text
    
    def test_placeholder_uniqueness(self, sample_text, mock_ner_pipeline, mock_streamlit):
        """Test che i placeholder siano unici"""
        anonymizer = NERAnonimizer()
        anonymizer._ner_pipe = mock_ner_pipeline
        
        anonymized_text, entities = anonymizer.anonymize(sample_text)
        
        # Tutti i placeholder dovrebbero essere unici
        placeholders = list(entities.keys())
        assert len(placeholders) == len(set(placeholders))
        
        # Ogni placeholder dovrebbe apparire nel testo
        for placeholder in placeholders:
            assert placeholder in anonymized_text

class TestAnonymizerEdgeCases:
    """Test casi limite"""
    
    def test_already_masked_text(self, mock_streamlit):
        """Test testo già parzialmente mascherato"""
        anonymizer = NERAnonimizer()
        
        text = "Contatta [EMAIL_0] per info su [CF_0]"
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        # Non dovrebbe ri-mascherare placeholder esistenti
        assert masked_text == text
        assert len(entities) == 0
    
    def test_overlapping_patterns(self, mock_streamlit):
        """Test pattern che si sovrappongono"""
        anonymizer = NERAnonimizer()
        
        # Testo con potenziali sovrapposizioni
        text = "Email test@domain.com nel sito https://test@domain.com"
        
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        # Dovrebbe gestire correttamente le sovrapposizioni
        assert len(entities) >= 1
        assert all(placeholder in masked_text for placeholder in entities.keys())
    
    def test_special_characters(self, mock_streamlit):
        """Test caratteri speciali"""
        anonymizer = NERAnonimizer()
        
        text = "Email: test@domain.com; IBAN: IT60X05428111010000001234567!"
        
        masked_text, entities = anonymizer.mask_with_regex(text)
        
        # Dovrebbe trovare entità anche con caratteri speciali intorno
        email_found = any('EMAIL' in k for k in entities.keys())
        iban_found = any('IBAN' in k for k in entities.keys())
        
        assert email_found
        assert iban_found