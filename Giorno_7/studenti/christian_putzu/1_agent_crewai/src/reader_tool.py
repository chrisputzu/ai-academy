import os
import glob
from typing import Dict, List

def read_documents(folder: str = "input_documents") -> List[Dict[str, str]]:
    """
    Read all files from a given folder and return their content and filenames.

    Parameters
    ----------
    folder : str, optional
        Path to the folder containing the input files. 
        Defaults to "input_documents".

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries, each containing:
        - 'text': the file content as a string.
        - 'file_name': the name of the file (without the full path).
    """
    docs: List[Dict[str, str]] = []
    for path in glob.glob(os.path.join(folder, "*")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            docs.append({
                "text": f.read(),
                "file_name": os.path.basename(path),
            })
    return docs
