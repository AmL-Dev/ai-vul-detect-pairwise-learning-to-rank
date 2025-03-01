import torch
import numpy as np
import random

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.utils import LOGGER
from src.data_prep import PairwiseDataset, SingletonDataset
from src.model import PairwiseRanker 
from src.train_eval import pairwise_loss

import numpy as np

def compute_pairwise_ltr_auc(vul_scores: np.ndarray, benign_scores: np.ndarray) -> float:
    """
    Compute the AUC (Area Under the Curve) using pairwise comparisons
    as per the Wilcoxon-Mann-Whitney (WMW) statistic.

    Parameters:
    - vul_scores: numpy array of positive point scores.
    - benign_scores: numpy array of negative point scores.

    Returns:
    - AUC value (float).
    """
    num_positives = len(vul_scores)
    num_negatives = len(benign_scores)

    # Count the number of correct pairwise orderings for all possible pairs
    correct_orderings = np.sum(vul_scores[:, None] > benign_scores[None, :])

    # Total number of pairs
    total_pairs = num_positives * num_negatives
    
    # Compute AUC
    auc = correct_orderings / total_pairs
    return auc

def compute_pairwise_ltr_indexed_accuracy(vul_scores: np.ndarray, benign_scores: np.ndarray) -> float:
    """
    Compute the accuracy by comparing scores at the same index
    between `vul_scores` and `benign_scores`.

    Parameters:
    - vul_scores: numpy array of positive point scores.
    - benign_scores: numpy array of negative point scores.

    Returns:
    - Accuracy (float).
    """
    # Ensure the arrays have the same length
    if len(vul_scores) != len(benign_scores):
        raise ValueError("Arrays must have the same length for indexed comparison.")
    
    # Compare pairs at the same index
    correct_count = np.sum(vul_scores > benign_scores)

    # Compute accuracy as the proportion of correct comparisons
    accuracy = correct_count / len(vul_scores)
    return accuracy


def calculate_pairwise_metrics(vul_scores: np.array, benign_scores: np.array, name_prefix: str) -> dict[str, float]:
    """
    Calculate model pairwise performance metrics based on vulnerable/benign pairs of model output scores.

    Args:
        vul_scores (np.array): model output scores for vulnerable code samples.
        benign_scores (np.array): model output scores for corresponding benign code samples.
        name_prefix (str): prefix appended before metric key.

    Returns:
        dict[str, float]: Dictionary of evaluation metrics and their values.  
    """

    # Compute accuracy
    accuracy = compute_pairwise_ltr_indexed_accuracy(vul_scores, benign_scores)
    # Compute AUC
    roc_auc = compute_pairwise_ltr_auc(vul_scores, benign_scores)

    result = {
        f"{name_prefix}_pairwise_acc": round(accuracy,4)*100,
        f"{name_prefix}_pairwise_roc_auc": round(roc_auc,4)*100
    }
    return result


def calculate_single_input_metrics(preds: np.array, labels: np.array, name_prefix: str) -> dict[str, float]:
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tpr = TP/(TP+FN)
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)
    roc_auc = roc_auc_score(labels, preds)

    result = {
        f"{name_prefix}_acc": round(acc,4)*100,
        f"{name_prefix}_prec": round(prec,4)*100,
        f"{name_prefix}_recall": round(recall,4)*100,
        f"{name_prefix}_f1": round(f1,4)*100,
        f"{name_prefix}_tpr": round(tpr,4)*100,
        f"{name_prefix}_tnr": round(tnr,4)*100,
        f"{name_prefix}_fpr": round(fpr,4)*100,
        f"{name_prefix}_fnr": round(fnr,4)*100,
        f"{name_prefix}_roc_auc": round(roc_auc, 4)*100
    }
    return result


def determine_optimal_threshold(
    model: PairwiseRanker,
    validation_data: DataLoader,
    name_prefix: str,
    thresholds: np.ndarray = np.arange(-0.5, 0, 0.01), 
) -> tuple[float, dict[str, float]]:
    """
    Determine the optimal threshold for binary classification based on validation data.

    Args:
        model (PairwiseRanker): The trained model.
        validation_data (DataLoader): Validation data with (code_snippet, label).
        tokenizer (AutoTokenizer): Tokenizer for encoding the code.
        device (torch.device): Device (CPU or GPU) to run the model.
        thresholds (np.ndarray): Array of thresholds to evaluate.
        name_prefix (str): prefix appended before metric key.

    Returns:
        Tuple[float, Dict[str, float]]: The optimal threshold and a dictionary of performance metrics.
    """
    model.eval()
    model = model.module if hasattr(model,'module') else model
    y_true, scores = [], []

    # Uses 10% of the validation dataset to find optimal threshold
    # Get total number of samples
    total_samples = len(validation_data.dataset)
    sample_size = int(total_samples * 0.1)
    # Create random indices for 10% of data
    random_indices = random.sample(range(total_samples), sample_size)
    subset_sampler = torch.utils.data.SubsetRandomSampler(random_indices)
    # Create a new DataLoader with the subset
    subset_validation_data = torch.utils.data.DataLoader(
        validation_data.dataset, 
        sampler=subset_sampler, 
        batch_size=validation_data.batch_size
    )

    # Compute scores for all validation samples
    with torch.no_grad():
        for batch in subset_validation_data:
            for code_snippet, label in zip(batch["func"], batch["label"]):
                with torch.no_grad():
                    # Pass pairs through the model
                    rank_score = model.compute_rank_score(code_snippet)
                    scores.append(rank_score)
                    y_true.append(label)

    # Evaluate metrics for each threshold
    best_threshold = None
    best_metric_scores = None
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in scores]

        metric_scores = calculate_single_input_metrics(y_pred, y_true, name_prefix)    

        # Update if F1-score improves
        if metric_scores[f"{name_prefix}_f1"] > best_f1:
            best_f1 = metric_scores[f"{name_prefix}_f1"]
            best_threshold = threshold
            best_metric_scores = metric_scores

    return best_threshold, best_metric_scores


def evaluate_model(
        model: PairwiseRanker, 
        eval_pairwise_dataloader: DataLoader, 
        eval_single_dataloader: DataLoader, 
        eval_class_threshold_single_dataloader: DataLoader, 
        device: torch.device, 
        name_prefix: str):
    """
    Evaluate model pairwise and single-input classification performance on the respective eval dataset
    """
    
    model.eval()
    
    ### Evaluate model for pairwise performance 
    # (i.e. ability to rank pairs correctly with score of vulnerable samples higher)
    eval_loss = 0.0
    nb_eval_steps = 0
    vul_scores=[] 
    benign_scores=[] 
    
    for batch in eval_pairwise_dataloader:
        code_vulnerable = batch["vulnerable_code"]
        code_benign = batch["benign_code"]
        # pairwise forward pass and loss
        with torch.no_grad():
            score_vuln, score_benign = model(code_vulnerable, code_benign)
            loss = pairwise_loss(score_vuln, score_benign)
            eval_loss += loss.mean().item()
            vul_scores.append(score_vuln.cpu().numpy())
            benign_scores.append(score_benign.cpu().numpy())
        nb_eval_steps += 1
    # Aggregate results from all batches
    vul_scores=np.concatenate(vul_scores,0)
    benign_scores=np.concatenate(benign_scores,0)
    # Evaluate results
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    pairwise_metrics = calculate_pairwise_metrics(vul_scores, benign_scores, name_prefix)
    
    
    ### Evaluate model for single input performance 
    # (i.e. ability to classify correctly sample as vulnerable or not)
    threshold = determine_optimal_threshold(model, eval_class_threshold_single_dataloader, name_prefix)[0]
    
    single_input_scores= []
    y_true = []
    
    model = model.module if hasattr(model,'module') else model

    for batch in eval_single_dataloader:
        inputs = batch["func"]     
        label=batch["label"]
        with torch.no_grad():
            rank_scores = model.compute_rank_score(inputs)
            single_input_scores.append(rank_scores.cpu().numpy())
            y_true.append(label.cpu().numpy())
        nb_eval_steps += 1

    single_input_scores=np.concatenate(single_input_scores).flatten()
    y_true=np.concatenate(y_true,0)
    
    # Apply threshold to classify
    y_preds = single_input_scores > threshold

    single_input_metrics = calculate_single_input_metrics(y_preds, y_true, name_prefix)    

    result = {f"{name_prefix}_loss": float(perplexity)}
    result.update(pairwise_metrics)
    result.update(single_input_metrics)
    return result


def validate(
        model: PairwiseRanker,
        primevul_paired_valid_data_file: str,
        primevul_single_input_valid_dataset: str,
        per_gpu_eval_batch_size: int,
        local_rank: int,
        n_gpu: int,
        device: torch.device,
        eval_when_training: bool = False):
    
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu) # Set batch size

    ### Load and batch the pairwise valid dataset
    # For both training and validation
    eval_pairwise_dataset = PairwiseDataset(primevul_paired_valid_data_file)
    eval_pairwise_sampler = RandomSampler(eval_pairwise_dataset) if local_rank == -1 else DistributedSampler(eval_pairwise_dataset) # Note that DistributedSampler samples randomly
    eval_pairwise_dataloader = DataLoader(eval_pairwise_dataset, sampler=eval_pairwise_sampler, batch_size=eval_batch_size)

    ### Load and batch the singleton valid dataset (for binary classification performance analysis)
    eval_singleton_dataset = SingletonDataset(primevul_single_input_valid_dataset)
    eval_singleton_sampler = RandomSampler(eval_singleton_dataset) if local_rank == -1 else DistributedSampler(eval_singleton_dataset) # Note that DistributedSampler samples randomly
    eval_singleton_dataloader = DataLoader(eval_singleton_dataset, sampler=eval_singleton_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    ### Evaluate the model
    LOGGER.info("***** Running evaluation *****")
    LOGGER.info("  Num pairwise examples = %d", len(eval_pairwise_dataset))
    LOGGER.info("  Num single input examples = %d", len(eval_singleton_dataset))
    LOGGER.info("  Batch size = %d", eval_batch_size)
    
    return evaluate_model(model, eval_pairwise_dataloader, eval_singleton_dataloader, eval_singleton_dataloader, device, "validation")


def test(
        model: PairwiseRanker,
        primevul_paired_test_data_file: str,
        primevul_single_input_test_dataset: str,
        primvul_singleton_valid_dataset_for_class_threshold: str,
        per_gpu_eval_batch_size: int,
        local_rank: int,
        n_gpu: int,
        device: torch.device):
    """
    Test trained model performance at pairwise ranking and at classifying single inputs.
    """
    
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    ### Load and batch the pairwise test dataset 
    eval_pairwise_dataset = PairwiseDataset(primevul_paired_test_data_file)
    eval_pairwise_sampler = SequentialSampler(eval_pairwise_dataset) if local_rank == -1 else DistributedSampler(eval_pairwise_dataset) # Note that DistributedSampler samples randomly
    eval_pairwise_dataloader = DataLoader(eval_pairwise_dataset, sampler=eval_pairwise_sampler, batch_size=eval_batch_size)

    ### Load and batch the singleton test dataset (for binary classification performance analysis)
    test_singleton_dataset = SingletonDataset(primevul_single_input_test_dataset)
    test_singleton_sampler = SequentialSampler(test_singleton_dataset) if local_rank == -1 else DistributedSampler(test_singleton_dataset) # Note that DistributedSampler samples randomly
    test_singleton_dataloader = DataLoader(test_singleton_dataset, sampler=test_singleton_sampler, batch_size=eval_batch_size)

    ### Load and batch the singleton valid dataset (to determine optimal rank score threshold classification )
    valid_singleton_dataset = SingletonDataset(primvul_singleton_valid_dataset_for_class_threshold)
    valid_singleton_sampler = SequentialSampler(valid_singleton_dataset) if local_rank == -1 else DistributedSampler(valid_singleton_dataset) # Note that DistributedSampler samples randomly
    valid_singleton_dataloader = DataLoader(valid_singleton_dataset, sampler=valid_singleton_sampler, batch_size=eval_batch_size)

    

    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    ### Evaluate the model
    LOGGER.info("***** Running Test *****")
    LOGGER.info("  Num pairwise examples = %d", len(eval_pairwise_dataset))
    LOGGER.info("  Num single input examples = %d", len(test_singleton_dataset))
    LOGGER.info("  Batch size = %d", eval_batch_size)
    
    return evaluate_model(model, eval_pairwise_dataloader, test_singleton_dataloader, valid_singleton_dataloader, device, "test")

