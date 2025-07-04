# 🧪 Test Suite

Suite di test automatici per il sistema di anonimizzazione documenti.

## 📋 Struttura Test

```
tests/
├── conftest.py              # Fixtures e configurazioni pytest
├── test_config.py           # Test configurazioni sistema
├── test_anonymizer.py       # Test anonimizzazione NER+Regex
├── test_ai_processor.py     # Test componenti AI (Azure+RAG+CrewAI)
├── test_utils.py            # Test funzioni utility
├── sample_data/             # Dati di test
└── README.md               # Questa documentazione
```

## 🚀 Come Eseguire i Test

### Setup Iniziale
```bash
# Installa dipendenze test
pip install -r requirements-test.txt

# Installa dipendenze principali
pip install -r requirements.txt
```

### Esecuzione Base
```bash
# Tutti i test
pytest

# Test specifico
pytest tests/test_anonymizer.py

# Test con coverage
pytest --cov

# Test veloci (escludi slow)
pytest -m "not slow"
```

### Esecuzione Avanzata
```bash
# Test in parallelo
pytest -n auto

# Test con output dettagliato
pytest -v

# Test solo falliti
pytest --lf

# Test con benchmark
pytest --benchmark-only
```

## 🏷️ Markers Disponibili

### **@pytest.mark.unit**
Test unitari veloci (<1s)
```bash
pytest -m unit
```

### **@pytest.mark.integration** 
Test di integrazione (<10s)
```bash
pytest -m integration
```

### **@pytest.mark.slow**
Test lenti (>10s)
```bash
pytest -m "not slow"  # Escludi
pytest -m slow        # Solo lenti
```

### **@pytest.mark.azure**
Test che richiedono Azure (con credenziali)
```bash
pytest -m "not azure"  # Senza Azure
```

## 🎯 Coverage Report

### Generazione Report
```bash
# HTML report
pytest --cov --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov --cov-report=term-missing

# XML report (per CI/CD)
pytest --cov --cov-report=xml
```

### Target Coverage
- **Minimo**: 80% overall
- **Obiettivo**: 90%+ per moduli core
- **Critico**: 95%+ per anonimizzazione

## 🧩 Test Categories

### **Unit Tests (80%)**
- Funzioni singole isolate
- Mock dipendenze esterne
- Execution rapida

### **Integration Tests (15%)**
- Componenti che interagiscono
- Mock servizi esterni (Azure)
- Execution media

### **End-to-End Tests (5%)**
- Workflow completi
- Test con UI Streamlit
- Execution lenta

## 📊 Test Data

### **Fixtures Disponibili**
- `sample_text`: Documento con entità varie
- `sample_text_no_entities`: Testo pulito
- `sample_entities`: Mappa entità esempio
- `mock_azure_client`: Client Azure mockato
- `mock_ner_pipeline`: Pipeline NER mockato

### **Sample Data**
```
tests/sample_data/
├── sample_document.txt         # Doc normale con entità
├── sample_with_entities.txt    # Doc ricco di entità
└── sample_empty.txt           # Doc vuoto
```

## 🔧 Configurazione

### **pytest.ini**
Configurazione pytest con:
- Markers personalizzati
- Coverage settings
- Warning filters
- Default options

### **conftest.py**
Fixtures condivise:
- Mock Azure OpenAI
- Mock Streamlit components
- Test data generators
- Environment setup

## 🐛 Debugging Test

### **Test Falliti**
```bash
# Re-run solo falliti
pytest --lf

# Stop al primo fallimento
pytest -x

# Debug mode
pytest --pdb
```

### **Test Specifici**
```bash
# Singolo test
pytest tests/test_anonymizer.py::TestNERAnonimizer::test_anonymize_complete_pipeline

# Classe di test
pytest tests/test_anonymizer.py::TestNERAnonimizer

# Con keyword
pytest -k "anonymize and not slow"
```

## 🚀 CI/CD Integration

### **GitHub Actions Example**
```yaml
- name: Run Tests
  run: |
    pytest --cov --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

### **Pre-commit Hooks**
```bash
# Installa pre-commit
pip install pre-commit
pre-commit install

# Test automatici prima del commit
pre-commit run --all-files
```

## 📈 Performance Testing

### **Benchmark Tests**
```bash
# Solo benchmark
pytest --benchmark-only

# Salva risultati
pytest --benchmark-save=baseline

# Confronta con baseline
pytest --benchmark-compare=baseline
```

### **Memory Testing**
```bash
# Con memory profiler
pytest --memray

# Test memory leaks
pytest --memray --trace-memory
```

## 🔍 Quality Checks

### **Code Quality**
```bash
# Linting
flake8 .

# Formatting
black --check .

# Import sorting
isort --check-only .
```

### **Security Scanning**
```bash
# Security vulnerabilities
bandit -r .

# Dependency check
safety check
```

## 📝 Writing New Tests

### **Naming Convention**
- File: `test_<module>.py`
- Class: `Test<ComponentName>`
- Method: `test_<what_it_tests>`

### **Test Structure**
```python
def test_function_behavior():
    # Arrange
    input_data = "test input"
    expected = "expected output"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected
```

### **Best Practices**
- ✅ Test one thing per test
- ✅ Use descriptive names
- ✅ Mock external dependencies
- ✅ Test edge cases
- ✅ Keep tests independent

## 🎪 Quick Commands

```bash
# Full test suite
make test

# Fast tests only
make test-fast

# Coverage report
make coverage

# Quality checks
make lint

# All checks
make check-all
```