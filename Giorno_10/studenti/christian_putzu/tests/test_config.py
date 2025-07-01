"""
Test per configurazioni sistema.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import os
from unittest.mock import patch
from config import Config, REGEX_PATTERNS

class TestConfig:
    """Test classe Config"""
    
    def test_config_attributes_exist(self):
        """Test che tutti gli attributi richiesti esistano"""
        assert hasattr(Config, 'NER_MODEL')
        assert hasattr(Config, 'AZURE_ENDPOINT')
        assert hasattr(Config, 'AZURE_API_KEY')
        assert hasattr(Config, 'DEPLOYMENT_NAME')
        
    def test_ner_model_default(self):
        """Test modello NER di default"""
        assert Config.NER_MODEL == "Davlan/bert-base-multilingual-cased-ner-hrl"
        
    def test_deployment_name_default(self):
        """Test deployment name di default"""
        assert Config.DEPLOYMENT_NAME == "gpt-4o"
        
    def test_api_version_default(self):
        """Test API version di default"""
        assert Config.AZURE_API_VERSION == "2024-02-01"
    
    @patch.dict(os.environ, {
        'AZURE_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_API_KEY': 'test-key'
    })
    def test_azure_config_from_env(self):
        """Test lettura configurazione da environment"""
        # Reload config per leggere nuove env vars
        import importlib
        import config
        importlib.reload(config)
        
        assert config.Config.AZURE_ENDPOINT == 'https://test.openai.azure.com/'
        assert config.Config.AZURE_API_KEY == 'test-key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_azure_config_missing(self):
        """Test configurazione Azure mancante"""
        import importlib
        import config
        importlib.reload(config)
        
        assert config.Config.AZURE_ENDPOINT is None
        assert config.Config.AZURE_API_KEY is None
    
    def test_openai_api_key_set(self):
        """Test che OPENAI_API_KEY sia settata se Azure disponibile"""
        with patch.dict(os.environ, {'AZURE_API_KEY': 'test-key'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert os.environ.get('OPENAI_API_KEY') == 'test-key'

class TestRegexPatterns:
    """Test pattern regex"""
    
    def test_regex_patterns_exist(self):
        """Test che tutti i pattern esistano"""
        required_patterns = ["IBAN", "EMAIL", "CF", "CARD", "PHONE"]
        
        for pattern in required_patterns:
            assert pattern in REGEX_PATTERNS
            assert isinstance(REGEX_PATTERNS[pattern], str)
            assert len(REGEX_PATTERNS[pattern]) > 0
    
    def test_iban_pattern(self):
        """Test pattern IBAN italiano"""
        import re
        pattern = re.compile(REGEX_PATTERNS["IBAN"])
        
        # IBAN valido
        assert pattern.search("IT60 X054 2811 1010 0000 0123 4567")
        assert pattern.search("IT60X05428111010000001234567")
        
        # IBAN invalido
        assert not pattern.search("GB60 X054 2811 1010 0000 0123 456")  # Non IT
        assert not pattern.search("IT60 X054")  # Troppo corto
    
    def test_email_pattern(self):
        """Test pattern email"""
        import re
        pattern = re.compile(REGEX_PATTERNS["EMAIL"])
        
        # Email valide
        assert pattern.search("test@example.com")
        assert pattern.search("user.name@domain.co.uk")
        assert pattern.search("test123@test-domain.org")
        
        # Email invalide
        assert not pattern.search("invalid-email")
        assert not pattern.search("@domain.com")
        assert not pattern.search("test@")
    
    def test_cf_pattern(self):
        """Test pattern codice fiscale"""
        import re
        pattern = re.compile(REGEX_PATTERNS["CF"])
        
        # CF valido (formato)
        assert pattern.search("RSSMRA80A01H501Z")
        assert pattern.search("VRDLCU85D15F205W")
        
        # CF invalido
        assert not pattern.search("RSSMRA80A01H501")  # Troppo corto
        assert not pattern.search("rssmra80a01h501z")  # Minuscolo
        assert not pattern.search("123456789012345")   # Solo numeri
    
    def test_card_pattern(self):
        """Test pattern carta di credito"""
        import re
        pattern = re.compile(REGEX_PATTERNS["CARD"])
        
        # Carte valide (formato)
        assert pattern.search("1234 5678 9012 3456")
        assert pattern.search("1234-5678-9012-3456")
        assert pattern.search("1234567890123456")
        
        # Carte invalide
        assert not pattern.search("1234 5678 9012")     # Troppo corto
        assert not pattern.search("abcd efgh ijkl mnop") # Lettere
    
    def test_phone_pattern(self):
        """Test pattern telefono"""
        import re
        pattern = re.compile(REGEX_PATTERNS["PHONE"])
        
        # Telefoni validi
        assert pattern.search("+39 333 1234567")
        assert pattern.search("333-123-4567")
        assert pattern.search("(02) 12345678")
        assert pattern.search("3331234567")
        
        # Telefoni invalidi
        assert not pattern.search("123")      # Troppo corto
        assert not pattern.search("abc-def")  # Lettere

class TestPatternValidation:
    """Test validazione pattern"""
    
    def test_all_patterns_compile(self):
        """Test che tutti i pattern si compilino correttamente"""
        import re
        
        for name, pattern in REGEX_PATTERNS.items():
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Pattern {name} non valido: {pattern}")
    
    def test_patterns_not_empty(self):
        """Test che nessun pattern sia vuoto"""
        for name, pattern in REGEX_PATTERNS.items():
            assert pattern.strip(), f"Pattern {name} è vuoto"
    
    def test_patterns_have_word_boundaries(self):
        """Test che i pattern usino word boundaries appropriati"""
        for name, pattern in REGEX_PATTERNS.items():
            # La maggior parte dei pattern dovrebbe avere \b per word boundary
            if name in ["IBAN", "CF", "CARD", "PHONE"]:
                assert "\\b" in pattern, f"Pattern {name} dovrebbe usare word boundaries"