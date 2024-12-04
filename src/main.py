#########################################################################
# Sign in to Hugging Face by running this command: huggingface-cli login
# 
# Set project location to be able to call project modules 
import sys
sys.path.append("/mnt/isgnas/home/anl31/documents/code/ai-vul-detect-pairwise-learning-to-rank")
#########################################################################

import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.utils import LOGGER
from src.utils import set_seed
from src.data_prep import PairwiseDataset
from src.data_prep import load_pairs_from_jsonl
from src.model import  PairwiseRanker
from src.train_eval import load_checkpoint
from src.train_eval import train_one_epoch
from src.train_eval import evaluate_model


def main(args):
    """
    Load data, train the model, evaluate results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================
    # Data Preparation
    # ============================
    
    train_pairs = load_pairs_from_jsonl(args.primevul_paired_train_data_file)
    valid_pairs = load_pairs_from_jsonl(args.primevul_paired_valid_data_file)
    test_pairs = load_pairs_from_jsonl(args.primevul_paired_test_data_file)

    train_dataset = PairwiseDataset(train_pairs)
    valid_dataset = PairwiseDataset(valid_pairs)
    test_dataset = PairwiseDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)


    # ============================
    # Load model and tokenizer
    # ============================

    # For distributed training 
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_embedder_name)
    encoder = AutoModel.from_pretrained(args.huggingface_embedder_name).to(device)

    model = PairwiseRanker(tokenizer, encoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    start_epoch = 0

    # Load checkpoint if resuming training
    if os.path.exists(args.checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        LOGGER.info(f"Reload model from {args.checkpoint_path}, resume training from epoch {start_epoch}.")
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


    # ============================
    # Training and Evaluation
    # ============================

    LOGGER.info("Training/evaluation parameters %s", args)

    # Training
    num_epochs = args.nb_epochs
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        valid_auc = evaluate_model(model, valid_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Validation AUC: {valid_auc:.4f}")

    # Final evaluation on test set
    test_auc = evaluate_model(model, test_loader)
    print(f"Test AUC: {test_auc:.4f}")


# Example usage
if __name__ == "__main__":
    # =====================
    # Parse input arguments
    # =====================

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Pairwise Learning to Rank Vulnerability Detection")
    
    # Commands related to general project setup
    parser.add_argument('--seed', type=int, default=42,
                        help="Optional random seed for initialization") # Optional
    parser.add_argument("--no_cuda", action='store_true',
                        help="Optional, avoid using CUDA when available") # Optional

    # Commands related to loading the dataset
    parser.add_argument("--primevul_paired_train_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired train data file.")
    parser.add_argument("--primevul_paired_valid_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired valid data file.")
    parser.add_argument("--primevul_paired_test_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired test data file.")
    
    # Commands related to the models
    parser.add_argument("--huggingface_embedder_name", required=True, type=str, 
                        help="Name of the Hugging Face model used for embedding (and tokenizing).")
    parser.add_argument("--checkpoint_path", default="", type=str, 
                        help="Optional, path to the model checkpoint to continue training.") # Optional
    parser.add_argument("--output_dir", required=True, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    # Commands related to training
    parser.add_argument("--do_train", action='store_true',
                        help="Optional, whether to run training.") # Optional
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Optional, For distributed training: local_rank") # Optional
    parser.add_argument("--train_batch_size", required=True, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", required=True, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--nb_epochs', type=int, required=True,
                        help="Number of training epochs.")

    # Parse arguments
    args = parser.parse_args()


    # ======================================
    # Setup CUDA, GPU & distributed training
    # ======================================
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu

    LOGGER.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    
    
    # Set seed
    set_seed(args.seed)    


    main(args)