import torch
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.utils import LOGGER
from src.data_prep import PairwiseDataset, SingletonDataset
from src.model import PairwiseRanker 
from src.train_eval import pairwise_loss

def calculate_pairwise_metrics(vul_scores: np.array, benign_scores: np.array) -> dict[str, float]:
    """
    Calculate model pairwise performance metrics based on vulnerable/benign pairs of model output scores.

    Args:
        pairwise_scores (list[tuple[float, float]]): vulnerable/benign pairs of model output scores.

    Returns:
        dict[str, float]: Dictionary of evaluation metrics and their values.  
    """

    # Compute accuracy
    correct_pairs = 0
    total_pairs = 0
    for score_vuln, score_benign in zip(vul_scores, benign_scores):
        if score_vuln > score_benign: # Check if pair is ranked correctly
            correct_pairs += 1
        total_pairs += 1
    accuracy = correct_pairs / total_pairs

    result = {"eval_pairwise_acc": round(accuracy,4)*100}
    return result


def calculate_single_input_metrics(preds: np.array, labels: np.array) -> dict[str, float]:
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
        "eval_acc": round(acc,4)*100,
        "eval_prec": round(prec,4)*100,
        "eval_recall": round(recall,4)*100,
        "eval_f1": round(f1,4)*100,
        "eval_tpr": round(tpr,4)*100,
        "eval_tnr": round(tnr,4)*100,
        "eval_fpr": round(fpr,4)*100,
        "eval_fnr": round(fnr,4)*100,
        "eval_roc_auc": round(roc_auc, 4)*100
    }
    return result


def determine_optimal_threshold(
    model: PairwiseRanker,
    validation_data: DataLoader,
    device: torch.device,
    thresholds: np.ndarray = np.arange(-0.5, 0.51, 0.01)
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
            for code_snippet, label in zip(batch["func"], batch["label"]):
                with torch.no_grad():
                    # Pass pairs through the model
                    rank_score = model.module.compute_rank_score(code_snippet)
                    scores.append(rank_score)
                    y_true.append(label)

    # Evaluate metrics for each threshold
    best_threshold = None
    best_metric_scores = None
    best_auc = -1.0

    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in scores]

        metric_scores = calculate_single_input_metrics(y_pred, y_true)    

        # Update if F1-score improves
        if metric_scores["eval_roc_auc"] > best_auc:
            best_auc = metric_scores["eval_roc_auc"]
            best_threshold = threshold
            best_metric_scores = metric_scores

    return best_threshold, best_metric_scores


def evaluate_model(model: PairwiseRanker, eval_pairwise_dataloader: DataLoader, eval_single_dataloader: DataLoader, eval_class_threshold_single_dataloader: DataLoader, device: torch.device):
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
    pairwise_metrics = calculate_pairwise_metrics(vul_scores, benign_scores)
    

    ### Evaluate model for single input performance 
    # (i.e. ability to classify correctly sample as vulnerable or not)
    threshold = determine_optimal_threshold(model, eval_class_threshold_single_dataloader, device)[0]
    
    single_input_scores= []
    y_true = []

    for batch in eval_single_dataloader:
        inputs = batch["func"]     
        label=batch["label"]
        with torch.no_grad():
            rank_scores = model.module.compute_rank_score(inputs)
            single_input_scores.append(rank_scores.cpu().numpy())
            y_true.append(label.cpu().numpy())
        nb_eval_steps += 1
    single_input_scores=np.concatenate(single_input_scores).flatten()
    y_true=np.concatenate(y_true,0)
            
    # Apply threshold to classify
    y_preds = single_input_scores > threshold

    single_input_metrics = calculate_single_input_metrics(y_preds, y_true)    

    result = {"eval_loss": float(perplexity)}
    result.update(pairwise_metrics)
    result.update(single_input_metrics)
    return result


def validate(
        model: PairwiseRanker,
        primevul_paired_valid_data_file: str,
        primevul_single_input_valid_dataset: str,
        eval_batch_size: int,
        local_rank: int,
        n_gpu: int,
        device: torch.device,
        eval_when_training: bool = False):

    ### Load and batch the pairwise valid dataset 
    eval_pairwise_dataset = PairwiseDataset(primevul_paired_valid_data_file)
    eval_pairwise_sampler = SequentialSampler(eval_pairwise_dataset) if local_rank == -1 else DistributedSampler(eval_pairwise_dataset) # Note that DistributedSampler samples randomly
    eval_pairwise_dataloader = DataLoader(eval_pairwise_dataset, sampler=eval_pairwise_sampler, batch_size=eval_batch_size)

    ### Load and batch the singleton valid dataset (for binary classification performance analysis)
    eval_singleton_dataset = SingletonDataset(primevul_single_input_valid_dataset)
    eval_singleton_sampler = SequentialSampler(eval_singleton_dataset) if local_rank == -1 else DistributedSampler(eval_singleton_dataset) # Note that DistributedSampler samples randomly
    eval_singleton_dataloader = DataLoader(eval_singleton_dataset, sampler=eval_singleton_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    ### Evaluate the model
    LOGGER.info("***** Running evaluation *****")
    LOGGER.info("  Num pairwise examples = %d", len(eval_pairwise_dataset))
    LOGGER.info("  Num single input examples = %d", len(eval_singleton_dataset))
    LOGGER.info("  Batch size = %d", eval_batch_size)
    
    return evaluate_model(model, eval_pairwise_dataloader, eval_singleton_dataloader, eval_singleton_dataloader, device)


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
    
    return evaluate_model(model, eval_pairwise_dataloader, test_singleton_dataloader, valid_singleton_dataloader, device)

