import ssl
import os
import json
import re
import openai

from typing import Union
from dotenv import load_dotenv
from src.ner_model import NERExtractor


class EmailAnonymizer:
    """
    EmailAnonymizer class for redacting sensitive information from text files using NER and GPT-4o.
    """

    def __init__(self, input_folder: str = "input_documents", output_folder: str = "output_documents") -> None:
        """
        Initialize the anonymizer, load environment variables and set up OpenAI and NER extractor.

        Parameters
        ----------
        input_folder : str
            Path to the folder containing input .txt files.
        output_folder : str
            Path to the folder where anonymized output will be saved.
        """

        # Confgure SSL certificate
        load_dotenv()

        # 1 STEP. Create output folder if it does not exist
        self.input_folder = input_folder
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
        else:
            print(f"Output folder already exists: {self.output_folder}")

        # 2 STEP. Configure environment and OpenAI client
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_api_key = os.getenv("AZURE_API_KEY")

        self.client = openai.AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-12-01-preview",
        )

        # 3 STEP. Load NER extractor
        self.extractor = NERExtractor()

    # 4 STEP. Create logger
    def log_debug(self, label: str, content: Union[str, dict, list]) -> None:
        """
        Print debug logs for inspection.

        Parameters
        ----------
        label : str
            Description label for the log output.
        content : str | dict | list
            The actual content to log.
        """
        print(f"\n[LOG] {label}\n{'-' * 60}")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2, ensure_ascii=False))
        else:
            print(content)

    # 5 STEP. Local Function to anonymize documents
    def anonymize_email(self, text: str, entities: dict) -> str:
        """
        Replace detected entities in text with placeholders.

        Parameters
        ----------
        text : str
            The original text.
        entities : dict
            Named entities with labels and values.

        Returns
        -------
        str
            Text with sensitive values replaced.
        """
        redacted_text = text
        for label, items in entities.items():
            for item in items:
                value = item if isinstance(item, str) else item.get("text", "")
                escaped_value = re.escape(value)
                placeholder = f"[{label}]"
                redacted_text = re.sub(escaped_value, placeholder, redacted_text)
        return redacted_text

    # 6 STEP. Iteration over all files in the input folder to do the following steps:
    def main(self) -> None:
        """
        Iteration over all files in the input folder to:

        1. Process documents in the input folder  
        2. Detect named entities  
        3. Create a flattened entity list  
        4. Anonymize the emails using the NERExtractor class  
        5. Generate a prompt to return anonymized emails using gpt-4o  
        6. Call the model with the prompt and return the response  
        7. Rename and save the output documents in the output folder
        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".txt"):
                print(f"\n[INFO] Processing file: {filename}")
                filepath = os.path.join(self.input_folder, filename)

                # 1. Process documents in the input folder
                with open(filepath, "r", encoding="utf-8") as fp:
                    text = fp.read()
                self.log_debug("Original Text (truncated)", text[:500] + ("..." if len(text) > 500 else ""))

                # 2. Detect named entities
                entities = self.extractor.extract_entities(text)
                self.log_debug("Detected Entities", entities)

                # 3. Create a flattened entity list
                flat_entities = [
                    (item if isinstance(item, str) else item.get("text"), label)
                    for label, items in entities.items()
                    for item in items
                ]
                self.log_debug("Flattened Entity List", flat_entities)

                if not flat_entities:
                    print("[INFO] No entities found. Skipping file.")
                    continue

                # 4. Anonymize the emails using the NERExtractor class
                anonymized_email = self.anonymize_email(text, entities)
                self.log_debug("Locally Anonymized Email (truncated)", anonymized_email[:500] + ("..." if len(anonymized_email) > 500 else ""))

                # 5. Generate a prompt to return anonymized emails using gpt-4o
                prompt = f"""
                You are a privacy assistant.\n"  # system preamble
                Below is a document that has already been anonymized with placeholders (e.g., [PER], [IBAN], [EMAIL]).\n
                "Review it and return only the fully anonymized version ensuring: \n
                No sensitive or personally identifiable information remains\n
                Placeholder format is consistent ([LABEL])\n
                You do not attempt to infer or reconstruct original data\n
                --- Document ---\n
                "{anonymized_email}"""
                self.log_debug("Prompt Sent to GPT-4o (truncated)", prompt[:500] + "...")

                try:
                    # 6. Call the model with the prompt and return the response
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a data-privacy assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=2024,
                        temperature=0.3
                    )

                    redacted_text = response.choices[0].message.content
                    self.log_debug("GPT-4o Response (truncated)", redacted_text[:500] + ("..." if len(redacted_text) > 500 else ""))

                    # 7. Rename and save the output documents in the output folder
                    base_filename = os.path.splitext(filename)[0]
                    output_filename = f"{base_filename}_anonymized.txt"
                    output_path = os.path.join(self.output_folder, output_filename)

                    with open(output_path, "w", encoding="utf-8") as out_fp:
                        out_fp.write(redacted_text)

                    print(f"[SUCCESS] File saved to: {output_path}")

                except Exception as e:
                    print(f"[ERROR] Failed to process {filename}: {e}")


if __name__ == "__main__":
    anonymizer = EmailAnonymizer()
    anonymizer.main()
