import os
import json
import re
import ssl
from typing import Union
from transformers import pipeline
from dotenv import load_dotenv
import openai

# SSL workaround (only if needed for custom certs)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["REQUESTS_CA_BUNDLE"] = r"C:\\Users\\SE645QY\\custom-ca-bundle.pem"

class NERExtractor:
    def __init__(self):
        self.ner_pipe = pipeline(
            task="token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device_map="auto"
        )

    def extract_entities(self, text: str) -> dict:
        result = {
            "PERSON": [],
            "ORG": [],
            "LOC": [],
            "DATE": [],
            "BANKCODE": [],
            "TAXCODE": [],
            "MISC": []
        }

        for ent in self.ner_pipe(text):
            tag = ent["entity_group"]
            word = ent["word"].strip()

            if tag == "PER":
                result["PERSON"].append(word)
            elif tag == "ORG":
                result["ORG"].append(word)
            elif tag == "LOC":
                result["LOC"].append(word)
            elif tag == "DATE":
                result["DATE"].append(word)
            elif tag == ["BANKCODE"]:
                result["BANKCODE"].append(word)
            elif tag == ["TAXCODE"]:
                result["TAXCODE"].append(word)
            else:
                result["MISC"].append(word)

        for key in result:
            seen = set()
            result[key] = [x for x in result[key] if not (x.lower() in seen or seen.add(x.lower()))]

        return result

class EmailAnonymizer:
    def __init__(self, input_folder: str = "input_documents", output_folder: str = "output_documents") -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
        else:
            print(f"Output folder already exists: {self.output_folder}")

        load_dotenv()
        azure_endpoint = os.getenv("AZURE_ENDPOINT") #inserire endpointe api key in questa sezione 
        azure_api_key = os.getenv("AZURE_API_KEY")

        self.client = openai.AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-12-01-preview",
        )

        self.extractor = NERExtractor()

    def log_debug(self, label: str, content: Union[str, dict, list]) -> None:
        print(f"\n[LOG] {label}\n{'-' * 60}")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2, ensure_ascii=False))
        else:
            print(content)

    def anonymize_email(self, text: str, entities: dict) -> str:
        redacted_text = text
        for label, items in entities.items():
            for item in items:
                value = item if isinstance(item, str) else item.get("text", "")
                escaped_value = re.escape(value)
                placeholder = f"[{label}]"
                redacted_text = re.sub(escaped_value, placeholder, redacted_text)
        return redacted_text

    def main(self) -> None:
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".txt"):
                print(f"\n[INFO] Processing file: {filename}")
                filepath = os.path.join(self.input_folder, filename)

                with open(filepath, "r", encoding="utf-8") as fp:
                    text = fp.read()
                self.log_debug("Original Text (truncated)", text[:500] + ("..." if len(text) > 500 else ""))

                entities = self.extractor.extract_entities(text)
                self.log_debug("Detected Entities", entities)

                flat_entities = [
                    (item if isinstance(item, str) else item.get("text"), label)
                    for label, items in entities.items()
                    for item in items
                ]
                self.log_debug("Flattened Entity List", flat_entities)

                if not flat_entities:
                    print("[INFO] No entities found. Skipping file.")
                    continue

                anonymized_email = self.anonymize_email(text, entities)
                self.log_debug("Locally Anonymized Email (truncated)", anonymized_email[:500] + ("..." if len(anonymized_email) > 500 else ""))

                prompt = f"""
                You are a privacy assistant helping to redact sensitive information from documents.
                Below is a document that has already been partially anonymized.  
                Use the list of named entities provided to ensure all sensitive data has been fully replaced with appropriate placeholders (e.g., [PERSON], [IBAN], [ORG], etc.).
                Do not attempt to reconstruct or guess any original information.
                --- Entities extracted ---
                {entities}
                --- Anonymized Document ---
                {anonymized_email}
                Please return only the fully anonymized version of the document.
                """
                self.log_debug("Prompt Sent to GPT-4o (truncated)", prompt[:500] + "...")

                try:
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
