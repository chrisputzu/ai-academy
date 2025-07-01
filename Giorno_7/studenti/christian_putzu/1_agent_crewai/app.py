import json
import os
from typing import Union, List, Dict
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai.llm import LLM

from src.reader_tool import read_documents
from src.ner_tool import NERExtractor
from src.anonimizer_tool import DocumentAnonymizer
from src.saver_tool import save_documents

# 1 STEP: Env variables configuration (to use gpt-4o)
load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# 2 STEP: LLm client configuration
llm_conf = LLM(
    model="azure/gpt-4o",
    api_version="2025-01-01-preview",
    api_key=azure_api_key,
    api_base=azure_endpoint,
)

# 3 STEP: Tools configuration
@tool("Document Loader Tool")
def load_documents_tool(folder: str = "input_documents") -> str:
    """
    Loads all text documents from the specified folder and returns a JSON-formatted list of dictionaries.

    Each dictionary contains:
      - 'text': the document content.
      - 'file_name': the original file name.

    Parameters
    ----------
    folder : str, optional
        The path to the input folder containing the documents (default is "input_documents").

    Returns
    -------
    str
        A JSON string representing a list of documents with their file names and contents.
    """
    docs = read_documents(folder)
    return json.dumps(docs, ensure_ascii=False)


@tool("NER Extraction Tool")
def extract_entities_tool(text: str) -> str:
    """
    Extracts named entities from the given text (e.g. names, IBANs, emails, etc.).

    Parameters
    ----------
    text : str
        The input text to analyze for named entities.

    Returns
    -------
    str
        A JSON string containing the extracted entities grouped by type.
    """
    ents = NERExtractor().extract_entities(text)
    return json.dumps(ents, ensure_ascii=False)


@tool("Anonymization Tool")
def anonymize_tool(text: str, entities_json: str) -> str:
    """
    Anonymizes the input text by replacing named entities with placeholder tags.

    Parameters
    ----------
    text : str
        The original input text containing sensitive information.
    entities_json : str
        A JSON string representing the named entities to be anonymized.

    Returns
    -------
    str
        The anonymized version of the input text with placeholders.
    """
    entities = json.loads(entities_json)
    return DocumentAnonymizer().anonymize(text, entities)


@tool("Document Saver Tool")
def save_documents_tool(anon_docs_json: Union[str, List[Dict[str, str]]], output_folder: str = "output_documents") -> str:
    """
    Saves anonymized documents into the specified output folder.

    Accepts a JSON string or a list of dictionaries with 'file_name' and 'text' fields.

    Parameters
    ----------
    anon_docs_json : Union[str, List[Dict[str, str]]]
        The anonymized documents, either as a JSON-formatted string or a list of dictionaries.
    output_folder : str, optional
        The target folder where files will be saved (default is "output_documents").

    Returns
    -------
    str
        A confirmation message indicating how many files were saved.
    """
    if isinstance(anon_docs_json, str):
        docs = json.loads(anon_docs_json)
    else:
        docs = anon_docs_json
    return save_documents(docs, output_folder)


# 4 STEP: Agents configuration
agent_loader = Agent(
    name="Document Loader",
    role="File System Specialist",
    goal="Read input documents and pass them to the pipeline.",
    backstory="Manages I/O operations in a robust and portable way across any OS.",
    tools=[load_documents_tool],
    llm=llm_conf,
)

agent_ner = Agent(
    name="Entity Recognizer",
    role="Data Analyst specialized in NER",
    goal="Detect all sensitive entities in the documents.",
    backstory=(
        "NLP expert in a privacy-centric team: accurately identifies any personal data."
    ),
    tools=[extract_entities_tool],
    llm=llm_conf,
)

agent_anonymizer = Agent(
    name="Document Anonymizer",
    role="Privacy Enforcement Specialist",
    goal="Replace detected personal data with labels like [NAME] or [EMAIL] in the original text without changing anything else.",
    backstory="Ensures GDPR-compliant anonymization by preserving the original structure of the document and replacing only the sensitive entities.",
    tools=[anonymize_tool],
    llm=llm_conf,
)


agent_saver = Agent(
    name="Document Saver",
    role="Output Handler",
    goal="Persist anonymized results using a consistent naming convention.",
    backstory="Ensures each file is saved correctly while avoiding naming conflicts.",
    tools=[save_documents_tool],
    llm=llm_conf,
)

# 5 STEP: Task configuration
task_load = Task(
    agent=agent_loader,
    description=(
        "Load all text documents from the 'input_documents' folder "
        "and return a JSON list with 'text' and 'file_name' fields."
    ),
    expected_output="JSON list of documents",
)

task_ner = Task(
    agent=agent_ner,
    context=[task_load],
    description=(
        "For each document in the above list, extract all names, "
        "IBANs, emails, organizations, and any other sensitive entities. \n\n"
        "Expected output: a JSON object with 'file_name' as key and a nested JSON "
        "containing the detected entities grouped by type as value."
    ),
    expected_output="Expected output: a JSON object with 'file_name' as key and a nested JSON "
"containing the detected entities grouped by type as value.",
)

task_anonymize = Task(
    agent=agent_anonymizer,
    context=[task_ner],
    description=(
        "Do NOT rewrite, reformat or paraphrase. "
        "Return exactly what the tool returns. Do not change or restructure the output."
    ),
    expected_output="JSON list of {'file_name', 'text'}",
)


task_save = Task(
    agent=agent_saver,
    context=[task_anonymize],
    description=(
        "For each document, write the original output from anoimize tool and save the documents "
        "into the 'output_documents' folder"
        "Confirm the number of files saved."
    ),
    expected_output="Confirmation of file saving",
)

# 6 STEP: Crew configuration
crew = Crew(
    agents=[agent_loader, agent_ner, agent_anonymizer, agent_saver],
    tasks=[task_load, task_ner, task_anonymize, task_save],
    process=Process.sequential,
    verbose=True,
)


if __name__ == "__main__":
    result = crew.kickoff()