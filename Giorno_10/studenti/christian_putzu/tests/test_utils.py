"""
Test per funzioni utility.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from utils import (
    validate_file_upload, export_results_json, get_confirmed_docs_count,
    add_chat_message, add_crewai_result, get_system_stats
)

class TestFileValidation:
    """Test validazione file"""
    
    def test_validate_file_upload_valid(self):
        """Test file valido"""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 1024  # 1KB
        
        assert validate_file_upload(mock_file) == True
    
    def test_validate_file_upload_none(self):
        """Test file None"""
        assert validate_file_upload(None) == False
    
    @patch('streamlit.error')
    def test_validate_file_upload_wrong_extension(self, mock_error):
        """Test estensione file sbagliata"""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        result = validate_file_upload(mock_file)
        
        assert result == False
        mock_error.assert_called_once()
    
    @patch('streamlit.error')
    def test_validate_file_upload_too_large(self, mock_error):
        """Test file troppo grande"""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 11 * 1024 * 1024  # 11MB
        
        result = validate_file_upload(mock_file)
        
        assert result == False
        mock_error.assert_called_once()

class TestExportResults:
    """Test export risultati"""
    
    def test_export_results_json_basic(self):
        """Test export JSON base"""
        data = {"test": "value", "number": 123}
        
        result = export_results_json(data, "test")
        
        # Verifica che sia JSON valido
        parsed = json.loads(result)
        assert parsed["test"] == "value"
        assert parsed["number"] == 123
        assert "metadata" in parsed
        assert "exported_at" in parsed["metadata"]
    
    def test_export_results_json_with_datetime(self):
        """Test export con datetime"""
        data = {"timestamp": datetime.now()}
        
        result = export_results_json(data, "test")
        
        # Non dovrebbe lanciare errori
        parsed = json.loads(result)
        assert "timestamp" in parsed
    
    def test_export_results_json_metadata(self):
        """Test metadati export"""
        data = {"item1": "value1", "item2": "value2"}
        
        result = export_results_json(data, "test")
        parsed = json.loads(result)
        
        assert "metadata" in parsed
        assert parsed["metadata"]["total_items"] == 2
        assert "exported_at" in parsed["metadata"]
        
        # Verifica formato ISO datetime
        timestamp = parsed["metadata"]["exported_at"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

class TestSessionStateHelpers:
    """Test helper per session state"""
    
    @patch('streamlit.session_state', {})
    def test_get_confirmed_docs_count_empty(self):
        """Test conteggio documenti confermati vuoto"""
        result = get_confirmed_docs_count()
        assert result == 0
    
    @patch('streamlit.session_state')
    def test_get_confirmed_docs_count_with_docs(self, mock_session):
        """Test conteggio documenti confermati"""
        mock_session.get.return_value = {
            'doc1': {'confirmed': True},
            'doc2': {'confirmed': False},
            'doc3': {'confirmed': True}
        }
        
        result = get_confirmed_docs_count()
        assert result == 2
    
    @patch('streamlit.session_state')
    def test_add_chat_message(self, mock_session):
        """Test aggiunta messaggio chat"""
        mock_session.chat_history = []
        
        add_chat_message("user", "Test message")
        
        assert len(mock_session.chat_history) == 1
        assert mock_session.chat_history[0]["role"] == "user"
        assert mock_session.chat_history[0]["content"] == "Test message"
    
    @patch('streamlit.session_state')
    def test_add_crewai_result(self, mock_session):
        """Test aggiunta risultato CrewAI"""
        mock_session.crewai_history = []
        
        add_crewai_result("test query", "comprehensive", "test result", ["agent1"])
        
        assert len(mock_session.crewai_history) == 1
        result = mock_session.crewai_history[0]
        
        assert result["query"] == "test query"
        assert result["analysis_type"] == "comprehensive"
        assert result["result"] == "test result"
        assert result["agents_used"] == ["agent1"]
        assert "timestamp" in result

class TestSystemStats:
    """Test statistiche sistema"""
    
    @patch('streamlit.session_state')
    def test_get_system_stats_empty(self, mock_session):
        """Test statistiche sistema vuoto"""
        mock_session.get.return_value = {}
        
        stats = get_system_stats()
        
        assert stats['uploaded_files'] == 0
        assert stats['anonymized_docs'] == 0
        assert stats['confirmed_docs'] == 0
        assert stats['processed_docs'] == 0
        assert stats['chat_messages'] == 0
        assert stats['crewai_analyses'] == 0
        assert stats['vector_store_ready'] == False
    
    @patch('streamlit.session_state')
    def test_get_system_stats_populated(self, mock_session):
        """Test statistiche sistema con dati"""
        def mock_get(key, default=None):
            data = {
                'uploaded_files': {'file1': {}, 'file2': {}},
                'anonymized_docs': {
                    'file1': {'confirmed': True},
                    'file2': {'confirmed': False}
                },
                'processed_docs': {'file1': {}},
                'chat_history': [{'role': 'user'}, {'role': 'assistant'}],
                'crewai_history': [{'query': 'test'}],
                'vector_store_built': True
            }
            return data.get(key, default)
        
        mock_session.get.side_effect = mock_get
        
        with patch('utils.get_confirmed_docs_count', return_value=1):
            stats = get_system_stats()
        
        assert stats['uploaded_files'] == 2
        assert stats['anonymized_docs'] == 2
        assert stats['confirmed_docs'] == 1
        assert stats['processed_docs'] == 1
        assert stats['chat_messages'] == 2
        assert stats['crewai_analyses'] == 1
        assert stats['vector_store_ready'] == True

class TestFileOperations:
    """Test operazioni file"""
    
    def test_temp_file_creation_and_cleanup(self, temp_test_file):
        """Test creazione e cleanup file temporaneo"""
        # File dovrebbe esistere
        assert os.path.exists(temp_test_file)
        
        # Contenuto dovrebbe essere corretto
        with open(temp_test_file, 'r') as f:
            content = f.read()
        assert content == "Test content for file operations"
        
        # Dopo il test, il file viene automaticamente rimosso dal fixture

class TestDataProcessing:
    """Test elaborazione dati"""
    
    def test_json_serialization_complex_data(self):
        """Test serializzazione dati complessi"""
        complex_data = {
            "string": "test",
            "number": 123,
            "float": 45.67,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {
                "inner": "value"
            },
            "datetime": datetime.now()
        }
        
        result = export_results_json(complex_data, "complex")
        
        # Dovrebbe serializzare senza errori
        parsed = json.loads(result)
        assert parsed["string"] == "test"
        assert parsed["number"] == 123
        assert parsed["float"] == 45.67
        assert parsed["boolean"] == True
        assert parsed["null"] is None
        assert parsed["list"] == [1, 2, 3]
        assert parsed["nested"]["inner"] == "value"
        assert "datetime" in parsed  # Convertito in stringa

class TestErrorHandling:
    """Test gestione errori"""
    
    @patch('streamlit.session_state', side_effect=Exception("Session state error"))
    def test_get_confirmed_docs_count_exception(self):
        """Test gestione eccezione in conteggio documenti"""
        # Dovrebbe gestire l'eccezione e tornare 0
        result = get_confirmed_docs_count()
        assert result == 0
    
    def test_export_results_with_non_serializable(self):
        """Test export con oggetti non serializzabili"""
        class NonSerializable:
            pass
        
        data = {"object": NonSerializable()}
        
        # Dovrebbe gestire oggetti non serializzabili
        result = export_results_json(data, "test")
        parsed = json.loads(result)
        
        # L'oggetto dovrebbe essere convertito in stringa
        assert "object" in parsed

class TestValidationHelpers:
    """Test helper di validazione"""
    
    def test_validate_file_upload_edge_cases(self):
        """Test casi limite validazione file"""
        # File con nome vuoto
        mock_file = Mock()
        mock_file.name = ""
        mock_file.size = 1024
        
        with patch('streamlit.error'):
            result = validate_file_upload(mock_file)
            assert result == False
        
        # File esattamente al limite (10MB)
        mock_file.name = "test.txt"
        mock_file.size = 10 * 1024 * 1024
        
        result = validate_file_upload(mock_file)
        assert result == True
        
        # File con estensione maiuscola
        mock_file.name = "test.TXT"
        mock_file.size = 1024
        
        result = validate_file_upload(mock_file)
        assert result == True

class TestIntegrationHelpers:
    """Test helper per integrazione"""
    
    @patch('streamlit.session_state')
    def test_session_state_integration(self, mock_session):
        """Test integrazione con session state"""
        # Simula stato iniziale
        mock_session.chat_history = []
        mock_session.crewai_history = []
        
        # Aggiungi dati
        add_chat_message("user", "Hello")
        add_chat_message("assistant", "Hi there")
        add_crewai_result("test query", "sentiment", "positive result")
        
        # Verifica stato finale
        assert len(mock_session.chat_history) == 2
        assert len(mock_session.crewai_history) == 1
        
        # Verifica contenuti
        assert mock_session.chat_history[0]["role"] == "user"
        assert mock_session.chat_history[1]["role"] == "assistant"
        assert mock_session.crewai_history[0]["analysis_type"] == "sentiment"