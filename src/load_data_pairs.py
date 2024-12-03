import json
from typing import List, Tuple

def load_pairs_from_jsonl(file_path: str) -> List[Tuple[str, str]]:
    """
    Extract pairs of vulnerable and benign code from a JSONL file for pairwise ranking.

    Args:
        file_path (str): Path to the JSONL file containing code snippets.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains:
            - The vulnerable code snippet (string)
            - The benign code snippet (string)
    """
    pairs = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate over lines in pairs (vulnerable followed by benign)
    for i in range(0, len(lines), 2):
        # Parse the current line and the next line
        vulnerable_entry = json.loads(lines[i])
        benign_entry = json.loads(lines[i + 1])

        # Validate the pair (ensure the target labels are correct)
        if vulnerable_entry['target'] == 1 and benign_entry['target'] == 0:
            pairs.append((vulnerable_entry['func'], benign_entry['func']))
        else:
            raise ValueError(f"Unexpected target labels for entries: {vulnerable_entry['target']}, {benign_entry['target']}")

    return pairs
