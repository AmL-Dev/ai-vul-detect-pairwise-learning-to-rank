import json
from dataclasses import dataclass
from typing import List
import logging
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch import Tensor

@dataclass
class CodePair:
    """Data class representing a pair of code snippets (vulnerable and benign)."""
    vulnerable: str
    benign: str
    vul_id: str
    vulnerable_hash: str
    benign_hash: str


def load_primevul_pairs(file_path: str) -> List[CodePair]:
        """Load and validate code pairs from a PrimeVul JSONL file."""
        logging.info(f"Loading PrimeVul pairs from {file_path}")
        pairs: List[CodePair] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_iter = iter(f)
                for vuln_line in line_iter:
                    benign_line = next(line_iter, None)
                    if benign_line is None:
                        break
                        
                    vuln = json.loads(vuln_line)
                    benign = json.loads(benign_line)
                    
                    pair = CodePair(
                        vulnerable=vuln.get('func', ''),
                        benign=benign.get('func', ''),
                        vul_id=str(vuln.get('vul_id', '')),
                        vulnerable_hash=str(vuln.get('func_hash', '')),
                        benign_hash=str(benign.get('func_hash', ''))
                    )
                    
                    if pair.vulnerable and pair.benign:
                        pairs.append(pair)
                    
            logging.info(f"Successfully loaded {len(pairs)} valid code pairs")
            return pairs
            
        except Exception as e:
            logging.error(f"Error loading pairs from {file_path}: {str(e)}")
            raise


class PairwiseCodeDataset(Dataset):
    """Dataset for pairwise learning of code vulnerability."""
    
    def __init__(
        self,
        pairs: List[CodePair],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        logging.info(f"Created dataset with {len(pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        pair = self.pairs[idx]
        
        # Tokenize both snippets
        pos_enc = self.tokenizer(
            pair.benign,  # benign is positive example
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        neg_enc = self.tokenizer(
            pair.vulnerable,  # vulnerable is negative example
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pos_input_ids": pos_enc.input_ids[0],
            "pos_attention_mask": pos_enc.attention_mask[0],
            "neg_input_ids": neg_enc.input_ids[0],
            "neg_attention_mask": neg_enc.attention_mask[0],
        }