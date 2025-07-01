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

    def __call__(self, text: str) -> dict:
        """
        Enable the class instance to be used as a callable (tool).
        
        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            Extracted named entities from the input text.
        """
        return self.extract_entities(text)


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

