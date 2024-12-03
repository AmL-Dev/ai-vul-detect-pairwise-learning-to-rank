"""All functions related to training the model
"""

import os
import torch

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


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return model, optimizer, start_epoch