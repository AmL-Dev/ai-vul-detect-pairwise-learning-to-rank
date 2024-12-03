import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

class PairwiseRanker(torch.nn.Module):
    """
    A pairwise ranking model using CodeBERT as the encoder.

    This model is designed for ranking code snippets based on their vulnerability.
    It encodes pairs of vulnerable and benign code snippets and computes a ranking score for each.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for the CodeBERT model.
        encoder (PreTrainedModel): Pretrained CodeBERT model for encoding code snippets.
        device (torch.device): Device (CPU or GPU) to run the model.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, encoder: PreTrainedModel, device: torch.device):
        super(PairwiseRanker, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = device
        self.fc = torch.nn.Linear(768, 1)  # CodeBERT outputs 768-dim embeddings
        self.activation = torch.nn.Tanh()  # Activation function to bound output to [-1, 1]


    def forward(self, code_vulnerable: list[str], code_benign: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute ranking scores for vulnerable and benign code snippets.

        Args:
            code_vulnerable (List[str]): Batch of vulnerable code snippets (list of strings).
            code_benign (List[str]): Batch of benign code snippets (list of strings).

        Returns:
            tuple: A tuple containing:
                - score_vuln (torch.Tensor): Ranking scores for vulnerable code snippets (shape: [batch_size, 1]).
                - score_benign (torch.Tensor): Ranking scores for benign code snippets (shape: [batch_size, 1]).
        """
        # Compute ranking scores
        score_vuln = self.compute_rank_score(code_vulnerable)
        score_benign = self.compute_rank_score(code_benign)
        
        return score_vuln, score_benign


    def compute_rank_score(self, code_batch: list[str]) -> torch.Tensor:
        """
        Compute compute ranking scores for input code snippets.

        Args:
            code_batch (List[str]): Batch of code snippets (list of strings).
        
        Returns:
            torch.Tensor: Ranking scores for input code snippets (shape: [batch_size, 1]).
        """
        # Encode the code snippets
        enc = self.encode(code_batch)

        # Compute ranking scores
        enc = self.fc(enc)
        score = self.activation(enc)

        return score


    def encode(self, code_batch: list[str]) -> torch.Tensor:
        """
        Encode a batch of code snippets using CodeBERT.

        Args:
            code_batch (List[str]): A batch of code snippets (list of strings).

        Returns:
            torch.Tensor: Encoded representations of the code snippets (shape: [batch_size, 768]).
        """
        # Tokenize the input batch
        inputs = self.tokenizer(code_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()} #Transform tokenized input tensors to appropriate device

        # Pass through the encoder
        outputs = self.encoder(**inputs).last_hidden_state

        # Use the [CLS] token embedding representation.
        # CodeBERT's [CLS] token (the first token of the sequence) is trained 
        # to capture global contextual information from the entire sequence.
        embeddings = outputs[:, 0, :]
        # embeddings = outputs.mean(dim=1)
        return embeddings