import os
from pathlib import Path
from typing import List, Dict

def save_documents(docs: List[Dict[str, str]], output_folder: str = "output_documents") -> str:
    """
    Save a list of anonymized documents to the specified output folder.

    Each file will be saved using its original filename and a tag _anonimized.

    Parameters
    ----------
    docs : List[Dict[str, str]]
        A list of dictionaries, each containing:
        - 'text': the anonymized text content.
        - 'file_name': the original file name (without path or extension).

    output_folder : str, optional
        The path to the folder where anonymized files will be saved.
        Defaults to "output_documents".

    Returns
    -------
    str
        A confirmation message indicating the number of files saved and the output folder path.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for doc in docs:
        out_path = os.path.join(
            output_folder, f"{doc['file_name']}_anonimized.txt"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])

    return f"Saved {len(docs)} files in '{output_folder}'"
