from torch.utils.data import Dataset
from src.data_prep import convert_examples_to_features

import json
import torch
from transformers import PreTrainedTokenizer

class PairwiseDataset(Dataset):
    """
    A PyTorch Dataset for pairwise ranking.
    Each item consists of:
        - Vulnerable code (positive)
        - Benign code (negative)
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        # Extract pairs of vulnerable and benign code from a JSONL file for pairwise ranking.
        pairs = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines), 2):
            vulnerable_entry = json.loads(lines[i])
            benign_entry = json.loads(lines[i + 1])
            if vulnerable_entry["target"] == 1 and benign_entry["target"] == 0:
                pairs.append((convert_examples_to_features(vulnerable_entry,tokenizer), convert_examples_to_features(benign_entry,tokenizer)))
            else:
                raise ValueError(f"Unexpected target labels: {vulnerable_entry['target']}, {benign_entry['target']}")
            
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        vulnerable_code, benign_code = self.pairs[i]
        return {"vulnerable_code": torch.tensor(vulnerable_code.input_ids), "benign_code": torch.tensor(benign_code.input_ids)}
    
    # def load_pairs_from_jsonl(self, file_path: str) -> list[tuple[str, str]]:
    #     """
    #     Extract pairs of vulnerable and benign code from a JSONL file for pairwise ranking.

    #     Args:
    #         file_path (str): Path to the JSONL file containing code snippets.

    #     Returns:
    #         List[Tuple[str, str]]: A list of tuples, where each tuple contains:
    #             - The vulnerable code snippet (string)
    #             - The benign code snippet (string)
    #     """
        
        