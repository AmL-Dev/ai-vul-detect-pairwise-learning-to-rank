import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from src.model import PairwiseRanker 

def evaluate_model(model: PairwiseRanker, dataloader: DataLoader):
    """
    Evaluate the model's pairwise ranking performance using accuracy.

    Args:
        model (PairwiseRanker): Trained PairwiseRanker model.
        dataloader (DataLoader): DataLoader containing pairs of vulnerable and benign samples.
    Returns:
        accuracy: Pairwise accuracy of the model.
    """
    model.eval()
    correct_pairs = 0
    total_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            for code_vulnerable, code_benign in zip(batch["vulnerable_code"], batch["benign_code"]):
                # Pass pairs through the model
                score_vuln, score_benign = model(code_vulnerable, code_benign)

                # Check if pair is ranked correctly
                if score_vuln > score_benign:
                    correct_pairs += 1
                total_pairs += 1

    accuracy = correct_pairs / total_pairs
    return accuracy



def determine_optimal_threshold(
    model: PairwiseRanker,
    validation_data: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    thresholds: np.ndarray = np.arange(-1.0, 1.01, 0.01)
) -> tuple[float, dict[str, float]]:
    """
    Determine the optimal threshold for binary classification based on validation data.

    Args:
        model (PairwiseRanker): The trained model.
        validation_data (DataLoader): Validation data with (code_snippet, label).
        tokenizer (AutoTokenizer): Tokenizer for encoding the code.
        device (torch.device): Device (CPU or GPU) to run the model.
        thresholds (np.ndarray): Array of thresholds to evaluate.

    Returns:
        Tuple[float, Dict[str, float]]: The optimal threshold and a dictionary of performance metrics.
    """
    model.eval()
    y_true, scores = [], []

    # Compute scores for all validation samples
    with torch.no_grad():
        for batch in validation_data:
            for code_vulnerable, code_benign in zip(batch["vulnerable_code"], batch["benign_code"]):
                # Pass pairs through the model
                score_vuln = model.compute_rank_score(code_vulnerable)
                scores.append(score_vuln)
                y_true.append(1)
                
                score_benign = model.compute_rank_score(code_benign)
                scores.append(score_benign)
                y_true.append(0)

    # Evaluate metrics for each threshold
    best_threshold = None
    best_metric_scores = None
    best_auc = -1.0

    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in scores]

        metric_scores = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred)
        }

        # Update if F1-score improves
        if metric_scores["roc_auc"] > best_auc:
            best_auc = metric_scores["roc_auc"]
            best_threshold = threshold
            best_metric_scores = metric_scores

    return best_threshold, best_metric_scores



def evaluate_single_input_classification(
    model: PairwiseRanker, 
    test_data: list[tuple[str, int]], 
    tokenizer: AutoTokenizer, 
    device: torch.device, 
    threshold: float = 0.0
) -> dict[str, float]:
    """
    Evaluate the performance of the binary classification task with only one input code snippet.

    Args:
        model (PairwiseRanker): The trained model.
        test_data (List[Tuple[str, int]]): A list of tuples where each tuple contains a code snippet 
                                           (str) and its ground truth label (int).
        tokenizer (AutoTokenizer): Tokenizer for encoding the code.
        device (torch.device): Device (CPU or GPU) to run the model.
        threshold (float): Decision threshold for classification (default=0.0).

    Returns:
        Dict[str, float]: Dictionary containing accuracy, F1-score, precision, and recall.
    """
    model.eval()
    y_true, y_pred = [], []

    for code_snippet, label in test_data:
        # Tokenize and encode the input code
        inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Encode the input code and calculate its score
            enc_code = model.encode(inputs["input_ids"], inputs["attention_mask"])
            score = model.fc(enc_code).item()

        # Apply threshold to classify
        predicted_label = 1 if score > threshold else 0

        # Store ground truth and prediction
        y_true.append(label)
        y_pred.append(predicted_label)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    
    return metrics


# def evaluate_model(model, dataloader):
#     """
#     Evaluate the model using AUC.
#     """
#     model.eval()
#     y_true, y_scores = [], []

#     with torch.no_grad():
#         for batch in dataloader:
#             code_vulnerable = batch["vulnerable_code"]
#             code_benign = batch["benign_code"]
            
#             score_vuln, score_benign = model(code_vulnerable, code_benign)
#             scores = (score_vuln - score_benign).squeeze().cpu().numpy()
#             y_true.extend([1] * len(score_vuln) + [0] * len(score_benign))
#             y_scores.extend(scores.tolist() + (-scores).tolist())
    
#     auc = roc_auc_score(y_true, y_scores)
#     return auc