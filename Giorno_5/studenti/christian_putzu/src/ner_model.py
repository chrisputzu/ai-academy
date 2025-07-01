import argparse
import json
import re
from transformers import pipeline

class NERExtractor:
    def __init__(self):
        """Initialize the NER pipeline using a multilingual NER model."""
        self.ner_pipe = pipeline(
            task="ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract_entities(self, text: str) -> dict:
        """
        Extract named entities from a given text using the NER pipeline and regex.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            A dictionary with keys representing entity types 
            (e.g., PER, ORG, LOC, DATE, IBAN, CODICE FISCALE, EMAIL, PHONE, MISC),
            and values as lists of unique detected entities.
        """
        result = {
            "PER": [],
            "ORG": [],
            "LOC": [],
            "DATE": [],
            "IBAN": [],
            "CODICE FISCALE": [],
            "EMAIL": [],
            "PHONE": [],
            "MISC": []
        }

        # NER model extraction
        for ent in self.ner_pipe(text):
            tag = ent["entity_group"]
            word = ent["word"].strip()
            if tag in result:
                result[tag].append(word)
            else:
                result["MISC"].append(word)

        # Regex-based extraction for (IBAN, CODICE FISCALE, EMAIL, PHONE, DATE)
        result["IBAN"].extend(re.findall(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b', text, flags=re.IGNORECASE))
        result["CODICE FISCALE"].extend(re.findall(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', text, flags=re.IGNORECASE))
        result["EMAIL"].extend(re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text))
        result["PHONE"].extend(re.findall(r'\+?\d[\d\s\-]{7,}\d', text))
        result["DATE"].extend(re.findall(r'\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}|(?:\d{1,2}\s)?(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s\d{4})\b', text, flags=re.IGNORECASE))

        # Deduplicate (case insensitive)
        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x.lower() in seen or seen.add(x.lower()))]

        return result

    def extract_from_file(self, file_path: str) -> dict:
        """
        Load text from a file and extract entities using the NER pipeline.

        Parameters
        ----------
        file_path : str
            Path to the text file containing input data.

        Returns
        -------
        dict
            Dictionary of extracted named entities grouped by type.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.extract_entities(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NER entities from a text file and output JSON.")
    parser.add_argument("document_path", nargs="?", default="document.txt", help="Path to the input document.txt file")
    args = parser.parse_args()
    extractor = NERExtractor()
    entities = extractor.extract_from_file(args.document_path)
    print(json.dumps(entities, ensure_ascii=False, indent=2))
