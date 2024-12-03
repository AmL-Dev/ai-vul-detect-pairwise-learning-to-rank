from torch.utils.data import Dataset

class PairwiseDataset(Dataset):
    """
    A PyTorch Dataset for pairwise ranking.
    Each item consists of:
        - Vulnerable code (positive)
        - Benign code (negative)
    """
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vulnerable_code, benign_code = self.pairs[idx]
        return {"vulnerable_code": vulnerable_code, "benign_code": benign_code}