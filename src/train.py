import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from typing import List, Tuple

# ============================
# Data Preparation
# ============================

class PairwiseDataset(Dataset):
    """
    A PyTorch Dataset for pairwise ranking.
    Each item consists of:
        - Vulnerable code (positive)
        - Benign code (negative)
    """
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vulnerable_code, benign_code = self.pairs[idx]
        return {"vulnerable_code": vulnerable_code, "benign_code": benign_code}


def load_pairs_from_jsonl(file_path: str) -> List[Tuple[str, str]]:
    """
    Extract pairs of vulnerable and benign code from a JSONL file for pairwise ranking.
    """
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
    return pairs


# ============================
# Model Definition
# ============================

class PairwiseRanker(torch.nn.Module):
    """
    A pairwise ranking model using CodeBERT as the encoder.
    """
    def __init__(self, encoder):
        super(PairwiseRanker, self).__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(768, 1)  # CodeBERT outputs 768-dim embeddings

    def forward(self, code_vulnerable, code_benign):
        # Encode vulnerable and benign code snippets
        enc_vuln = self.encode(code_vulnerable)
        enc_benign = self.encode(code_benign)
        
        # Compute pairwise ranking score
        score_vuln = self.fc(enc_vuln)
        score_benign = self.fc(enc_benign)
        
        return score_vuln, score_benign

    def encode(self, code_batch):
        # Tokenize and encode a batch of code snippets
        inputs = tokenizer(code_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()} #Transform tokenized input tensors to appropriate device
        outputs = self.encoder(**inputs).last_hidden_state
        return outputs.mean(dim=1)


# ============================
# Training and Evaluation
# ============================

def pairwise_loss(score_vuln, score_benign):
    """
    Pairwise ranking loss: logistic loss.
    """
    return torch.nn.functional.logsigmoid(score_vuln - score_benign).mean()


def train_one_epoch(model, dataloader, optimizer, epoch, save_path):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        code_vulnerable = batch["vulnerable_code"]
        code_benign = batch["benign_code"]
        
        score_vuln, score_benign = model(code_vulnerable, code_benign)
        loss = pairwise_loss(score_vuln, score_benign)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Save model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_path, f"checkpoint_epoch_{epoch}.pt"))

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader):
    """
    Evaluate the model using AUC.
    """
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for batch in dataloader:
            code_vulnerable = batch["vulnerable_code"]
            code_benign = batch["benign_code"]
            
            score_vuln, score_benign = model(code_vulnerable, code_benign)
            scores = (score_vuln - score_benign).squeeze().cpu().numpy()
            y_true.extend([1] * len(score_vuln) + [0] * len(score_benign))
            y_scores.extend(scores.tolist() + (-scores).tolist())
    
    auc = roc_auc_score(y_true, y_scores)
    return auc


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return model, optimizer, start_epoch


# ============================
# Main Training Loop
# ============================

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    encoder = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    save_path = "/model_checkpoints"
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare datasets
    primevul_paired_train_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
    primevul_paired_valid_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired.jsonl"
    primevul_paired_test_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"

    train_pairs = load_pairs_from_jsonl(primevul_paired_train_data_file)
    valid_pairs = load_pairs_from_jsonl(primevul_paired_valid_data_file)
    test_pairs = load_pairs_from_jsonl(primevul_paired_test_data_file)

    train_dataset = PairwiseDataset(train_pairs)
    valid_dataset = PairwiseDataset(valid_pairs)
    test_dataset = PairwiseDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = PairwiseRanker(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Load checkpoint if resuming training
    checkpoint_path = "path_to_checkpoint.pt"  # Replace with actual path if resuming
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resumed training from epoch {start_epoch}.")

    # Training
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, save_path)
        valid_auc = evaluate_model(model, valid_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Validation AUC: {valid_auc:.4f}")

    # Final evaluation on test set
    test_auc = evaluate_model(model, test_loader)
    print(f"Test AUC: {test_auc:.4f}")
