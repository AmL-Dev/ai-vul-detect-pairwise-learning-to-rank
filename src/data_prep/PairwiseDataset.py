from torch.utils.data import Dataset
import json

class PairwiseDataset(Dataset):
    """
    A PyTorch Dataset for pairwise ranking.
    Each item consists of:
        - Vulnerable code (positive)
        - Benign code (negative)
    """
    def __init__(self, file_path: str):
        # Extract pairs of vulnerable and benign code from a JSONL file for pairwise ranking.
        pairs = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines), 2):
            vulnerable_entry = json.loads(lines[i])
            benign_entry = json.loads(lines[i + 1])
            if vulnerable_entry["target"] == 1 and benign_entry["target"] == 0:
                pairs.append((vulnerable_entry["func"], benign_entry["func"]))
            else:
                raise ValueError(f"Unexpected target labels: {vulnerable_entry['target']}, {benign_entry['target']}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vulnerable_code, benign_code = self.pairs[idx]
        return {"vulnerable_code": vulnerable_code, "benign_code": benign_code}