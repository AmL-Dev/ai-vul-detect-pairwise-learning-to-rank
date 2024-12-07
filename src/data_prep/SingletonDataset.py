import json
from transformers import PreTrainedTokenizer

from torch.utils.data import Dataset

class SingletonDataset(Dataset):
    """
    A PyTorch Dataset for holding code snippet and label values from the PrimeVul dataset.
    Each item consists of:
        - func (code snippet)
        - label (vulnerable or benign)
    """
    def __init__(self, file_path: str):
        self.singletons = self.load_singletons_from_jsonl(file_path)

    def __len__(self):
        return len(self.singletons)

    def __getitem__(self, i):
        code_snippet, label = self.singletons[i]
        return {"func": code_snippet, "label": label}
    
    def load_singletons_from_jsonl(self, file_path: str) -> list[tuple[str, int]]:
        """
        Extract singletons of code (vulnerable or benign) from a JSONL file.

        Args:
            file_path (str): Path to the JSONL file containing code snippets.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains:
                - The code snippet (string)
                - The binary label (int)
        """
        singletons = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            code_snippet = json.loads(line)
            singletons.append((code_snippet["func"], code_snippet["target"]))
        return singletons