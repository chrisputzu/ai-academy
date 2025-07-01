"""
Configurazioni pytest e fixtures condivise.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Aggiungi src al path per import
sys.path.insert(0, r"Giorno_10\src")

@pytest.fixture
def sample_text():
    """Testo di esempio per test"""
    return """
    Gentile Mario Rossi,
    
    La contatto in merito alla fattura n. 12345.
    Il suo codice fiscale RSSMRA80A01H501Z risulta corretto.
    
    Per il pagamento può utilizzare:
    IBAN: IT60 X054 2811 1010 0000 0123 456
    Email: mario.rossi@example.com
    Telefono: +39 333 1234567
    
    Carta: 4532 1234 5678 9012
    
    Cordiali saluti,
    Ufficio Amministrazione
    ACME SpA
    """

@pytest.fixture
def sample_text_no_entities():
    """Testo senza entità sensibili"""
    return """
    Questo è un documento di prova
    che non contiene informazioni sensibili.
    
    Solo testo normale per i test.
    """

@pytest.fixture
def sample_empty_text():
    """Testo vuoto"""
    return ""

@pytest.fixture
def sample_entities():
    """Entità di esempio per test"""
    return {
        "[PER_0]": "Mario Rossi",
        "[CF_0]": "RSSMRA80A01H501Z",
        "[IBAN_0]": "IT60 X054 2811 1010 0000 0123 456",
        "[EMAIL_0]": "mario.rossi@example.com",
        "[PHONE_0]": "+39 333 1234567",
        "[CARD_0]": "4532 1234 5678 9012",
        "[ORG_0]": "ACME SpA"
    }

@pytest.fixture
def mock_azure_config():
    """Mock configurazioni Azure"""
    with patch.dict(os.environ, {
        'AZURE_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_API_KEY': 'test-api-key',
        'AZURE_ENDPOINT_EMB': 'https://test-emb.openai.azure.com/',
        'AZURE_API_KEY_EMB': 'test-emb-key'
    }):
        yield

@pytest.fixture
def mock_azure_client():
    """Mock client Azure OpenAI"""
    mock_client = Mock()
    
    # Mock response per chat completion
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test analysis result"
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_ner_pipeline():
    """Mock pipeline NER"""
    mock_pipeline = Mock()
    
    # Mock entità rilevate
    mock_entities = [
        {
            'entity_group': 'PER',
            'score': 0.9,
            'start': 8,
            'end': 19,
            'word': 'Mario Rossi'
        },
        {
            'entity_group': 'ORG', 
            'score': 0.8,
            'start': 200,
            'end': 208,
            'word': 'ACME SpA'
        }
    ]
    
    mock_pipeline.return_value = mock_entities
    return mock_pipeline

@pytest.fixture
def temp_test_file():
    """File temporaneo per test"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for file operations")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def mock_streamlit():
    """Mock componenti Streamlit per test"""
    with patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.spinner') as mock_spinner:
        
        # Spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)
        
        yield {
            'error': mock_error,
            'warning': mock_warning, 
            'success': mock_success,
            'info': mock_info,
            'spinner': mock_spinner
        }

@pytest.fixture
def sample_anonymized_docs():
    """Documenti anonimizzati di esempio"""
    return {
        'document1.txt': {
            'original': 'Documento con Mario Rossi e mario@email.com',
            'anonymized': 'Documento con [PER_0] e [EMAIL_0]',
            'entities': {
                '[PER_0]': 'Mario Rossi',
                '[EMAIL_0]': 'mario@email.com'
            },
            'confirmed': True
        },
        'document2.txt': {
            'original': 'Altro documento con ACME SpA',
            'anonymized': 'Altro documento con [ORG_0]',
            'entities': {
                '[ORG_0]': 'ACME SpA'
            },
            'confirmed': False
        }
    }

# Configurazioni pytest
def pytest_configure(config):
    """Configurazione pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "azure: marks tests that require Azure credentials"
    )