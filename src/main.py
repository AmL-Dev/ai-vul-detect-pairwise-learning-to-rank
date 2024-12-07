#########################################################################
# Sign in to Hugging Face by running this command: huggingface-cli login
# 
# Set project location to be able to call project modules 
import sys
sys.path.append("/mnt/isgnas/home/anl31/documents/code/ai-vul-detect-pairwise-learning-to-rank")

# TODO: Remove this and put tokenizer in PairwiseDataset 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#########################################################################

import argparse
import torch
from transformers import AutoTokenizer, AutoModel

from src.utils import LOGGER
from src.utils import set_seed
from src.data_prep import PairwiseDataset
from src.model import  PairwiseRanker
from src.train_eval import load_checkpoint
from src.train_eval import train
from src.train_eval import test


def main(args):
    """
    Load data, train the model, evaluate results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # ============================
    # Load model and tokenizer
    # ============================

    # For distributed training 
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_embedder_name)
    encoder = AutoModel.from_pretrained(args.huggingface_embedder_name).to(args.device)

    model = PairwiseRanker(tokenizer, encoder, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    args.start_epoch = 0
    # TODO Check the use of this var
    args.start_step = 0 

    # Load checkpoint if resuming training
    if os.path.exists(args.checkpoint_path):
        model, optimizer, args.start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        LOGGER.info(f"Reload model from {args.checkpoint_path}, resume training from epoch {args.start_epoch}.")
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


    # ============================
    # Training
    # ============================
    LOGGER.info("Training/evaluation parameters %s", args)

    if args.do_train:
        # Load training data
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_dataset = PairwiseDataset(args.primevul_paired_train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()
        
        train(args, train_dataset, model)
        
    # ============================
    # Testing
    # ============================
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        result=test(
            model,
            args.primevul_paired_test_data_file,
            args.primevul_single_input_test_dataset,
            args.primevul_single_input_valid_dataset,
            args.per_gpu_eval_batch_size,
            args.local_rank,
            args.n_gpu,
            args.device)
        LOGGER.info("***** Test results *****")
        for key in sorted(result.keys()):
            LOGGER.info("  %s = %s", key, str(round(result[key],4)))


if __name__ == "__main__":
    # =====================
    # Parse input arguments
    # =====================

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Pairwise Learning to Rank Vulnerability Detection")
    
    # Commands related to general project setup
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization") # Optional
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") # Optional

    # Commands related to loading the dataset
    parser.add_argument("--primevul_paired_train_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired train data file.")
    parser.add_argument("--primevul_paired_valid_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired valid data file.")
    parser.add_argument("--primevul_paired_test_data_file", required=True, type=str, 
                        help="Path to the PrimeVul paired test data file.")
    parser.add_argument("--primevul_single_input_valid_dataset", required=True, type=str, 
                        help="Path to the PrimeVul non-paired valid data file.")
    parser.add_argument("--primevul_single_input_test_dataset", required=True, type=str, 
                        help="Path to the PrimeVul non-paired test data file.")
    
    # Commands related to the models
    parser.add_argument("--huggingface_embedder_name", required=True, type=str, 
                        help="Name of the Hugging Face model used for embedding (and tokenizing).")
    parser.add_argument("--checkpoint_path", default="", type=str, 
                        help="Path to the model checkpoint to continue training.") # Optional
    
    # Commands related to training 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.") # Optional
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.") # Optional
    parser.add_argument("--max_patience", type=int, default=100,
                        help="Number of epochs after which stop training if the model did not improve.") # Optional
    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank") # Optional
    # training hyperparameters
    parser.add_argument("--learning_rate", required=True, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", required=True, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", required=True, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--nb_epochs', type=int, required=True,
                        help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm (to prevent exploding gradients).") # Optional
    # Track results and training stats
    parser.add_argument("--output_dir", required=True, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--run_dir', type=str, default="runs", 
                        help="Parent directory to store run stats.") # Optional
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.") # Optional
    

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
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))
    
    
    # Set seed
    set_seed(args.seed)    


    main(args)