import re
from typing import Dict, List

class DocumentAnonymizer:
    """
    Anonymizes a given text by replacing specific spans corresponding to detected entities
    with their entity type enclosed in square brackets (e.g., [NAME], [EMAIL], [IBAN]).

    The replacement preserves the original structure of the text and avoids altering
    substrings within other words by using word boundaries in the regular expressions.
    """

    def anonymize(self, text: str, entities: Dict[str, List[str]]) -> str:
        """
        Replace all occurrences of detected entity spans in the input text with their
        corresponding labels in square brackets. Replacement is case-insensitive and avoids
        partial word matches.

        Parameters
        ----------
        text : str
            The original input text to be anonymized.
        entities : Dict[str, List[str]]
            A dictionary where keys are entity types (e.g., 'PER', 'EMAIL')
            and values are lists of detected entity strings to be replaced.

        Returns
        -------
        str
            The anonymized text with entity spans replaced by their labels.
        """
        if not entities:
            return text

        spans = [
            (span, label) for label, values in entities.items() for span in values if span
        ]
        spans.sort(key=lambda p: len(p[0]), reverse=True)

        for span_text, label in spans:
            escaped = re.escape(span_text)
            pattern = rf'\b{escaped}\b'
            replacement = f'[{label.upper()}]'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text
