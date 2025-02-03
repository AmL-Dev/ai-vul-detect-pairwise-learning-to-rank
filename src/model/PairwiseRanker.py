import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

class PairwiseRanker(nn.Module):
    """
    A pairwise ranking model using CodeBERT as the encoder.

    This model is designed for ranking code snippets based on their vulnerability.
    It encodes pairs of vulnerable and benign code snippets and computes a ranking score for each.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for the CodeBERT model.
        encoder (PreTrainedModel): Pretrained CodeBERT model for encoding code snippets.
        device (torch.device): Device (CPU or GPU) to run the model.
    """
    def __init__(self, tokenizer: AutoTokenizer, encoder: AutoModel, device: torch.device):
        super(PairwiseRanker, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = device
        
        # Retrieve max length from the model's configuration
        
        if not hasattr(self.encoder.config, "max_position_embeddings"):
            self.encoder.config.max_position_embeddings = 16384
        self.max_length = self.encoder.config.max_position_embeddings - 2  # Reserve space for special tokens

        self.dim_encoder_output = encoder.config.hidden_size # embeddings outputs dim 
        
        ### NO embedder ablation study
        # self.tokenizer_vocab_size = tokenizer.vocab_size # Define vocab size
        
        hidden_dim = 512  # Hidden layer size
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.dim_encoder_output),
            nn.ReLU() ,
            nn.Linear(self.dim_encoder_output, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU() ,
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 1),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )

        # Initialize weights with Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # He initialization for weights
                nn.init.zeros_(layer.bias)            # Zero initialization for biases


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
        score = self.layers(enc)
        return score


    def encode(self, code_batch: list[str]) -> torch.Tensor:
        """
        Encode a batch of code snippets using CodeBERT.

        Args:
            code_batch (List[str]): A batch of code snippets (list of strings).

        Returns:
            torch.Tensor: Encoded representations of the code snippets (shape: [batch_size, 768]).
        """
        with torch.no_grad():
            # ### CodeT5
            # # Tokenize the text batch with padding and truncation for consistent input sizes
            # encoded_input = self.tokenizer(
            #     code_batch,
            #     padding=True,
            #     truncation=True,
            #     return_tensors="pt"
            # ).to(self.device)

            # # Use only the encoder's output for embedding extraction
            # with torch.no_grad():
            #     encoder_outputs = self.encoder.encoder(**encoded_input)

            # # Use the `last_hidden_state` from the encoder's output
            # embeddings = encoder_outputs.last_hidden_state.mean(dim=1)

            # return embeddings



            # ### dunzhang/stella_en_400M_v5

            # # Tokenize the input batch and truncate to the maximum length accepted by the mode
            # tokenized_input = self.tokenizer(code_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=None)
            # tokenized_input = {k: v.to(self.device) for k, v in tokenized_input.items()} #Transform tokenized input tensors to appropriate device

            # attention_mask = tokenized_input["attention_mask"]
            
            # # Pass through the encoder
            # last_hidden_state = self.encoder(**tokenized_input)[0]

            # last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            # embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            # return embeddings



            ### CodeBert, Unixcoder-base

            # Tokenize the input batch
            tokenized_input = self.tokenizer(code_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            tokenized_input = {k: v.to(self.device) for k, v in tokenized_input.items()} #Transform tokenized input tensors to appropriate device
            # ### NO embedder ablation study
            # input_ids = tokenized_input["input_ids"]  # Shape: [batch_size, sequence_length]
            # # One-hot encode token IDs
            # one_hot = self.one_hot = nn.functional.one_hot(input_ids, num_classes=self.tokenizer_vocab_size).float()  # Shape: [batch_size, sequence_length, vocab_size]
            # # # Sum over the sequence dimension to get a bag-of-tokens representation
            # # bag_of_tokens = one_hot.sum(dim=1)  # Shape: [batch_size, vocab_size]
            # # return bag_of_tokens
            
            # # Apply mean pooling
            # pooled_output = one_hot.mean(dim=1)  # Shape: [batch_size, vocab_size]
            # return pooled_output


            ### WITH embedder
            
            # Pass through the encoder
            outputs = self.encoder(**tokenized_input).last_hidden_state

            # Use the [CLS] token embedding representation (ensure that the model has [CLS] token).
            # CodeBERT's [CLS] token (the first token of the sequence) is trained 
            # to capture global contextual information from the entire sequence.
            # embeddings = outputs[:, 0, :]

            ### Mean pooling across tokens
            # Mask to ignore padding tokens during pooling
            attention_mask = tokenized_input["attention_mask"]  # Shape: (batch_size, sequence_length)
            mask = attention_mask.unsqueeze(-1).expand(outputs.size())  # Shape: (batch_size, sequence_length, hidden_size)
            masked_outputs = outputs * mask  # Zero out padding token embeddings

            # Sum the embeddings and divide by the number of non-padded tokens
            masked_outputs = outputs * mask  # Zero out padding token embeddings
            embeddings = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, hidden_size)
            
            # embeddings = outputs.mean(dim=1)
            
            return embeddings