"""All functions related to training the model
"""

import os
import torch
import wandb

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return model, optimizer, start_epoch



from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from src.utils import LOGGER
from src.model import  PairwiseRanker
from src.data_prep import PairwiseDataset
from src.train_eval import validate, pairwise_loss, calculate_pairwise_metrics


def train_one_epoch(model: PairwiseRanker, train_dataloader: DataLoader, optimizer, args, epoch: int):
    """
    Train the model for one epoch.

    Args:
        model: The model being trained.
        train_dataloader: DataLoader for training data.
        optimizer: Optimizer for gradient updates.
        args: Training arguments (assumed to have attributes like n_gpu, max_grad_norm, log_to_wandb).
        epoch: Current epoch.

    Returns:
        avg_loss: Average loss for the epoch.
        metrics: Dictionary of training metrics.
    """
    LOGGER.info(f"***** Training epoch {epoch} *****")
    LOGGER.info("  Num iteration = %d", len(train_dataloader))

    model.train()
    train_loss_list = []
    vul_scores_list = []
    benign_scores_list = []

    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    for local_step, batch in enumerate(bar):
        code_vulnerable = batch["vulnerable_code"]
        code_benign = batch["benign_code"]

        # Forward pass
        score_vuln, score_benign = model(code_vulnerable, code_benign)

        # Calculate loss
        loss = pairwise_loss(score_vuln, score_benign)
        if args.n_gpu > 1:
            loss = loss.mean()  # Average loss across GPUs

        # Backpropagation
        optimizer.zero_grad() # Clear gradients for next train
        loss.backward() # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # Prevent exploding gradients
        optimizer.step() # Apply gradients

        # Update training stats
        train_loss_list.append(loss.item())
        vul_scores_list.append(score_vuln.detach().cpu().numpy())
        benign_scores_list.append(score_benign.detach().cpu().numpy())
        bar.set_description(f"Epoch {epoch}, Epoch Step {local_step}, Loss: {loss.item():.4f}")

    # Compute average loss
    median_loss = np.median(train_loss_list)

    # Concatenate scores for metric calculation
    vul_scores = np.concatenate(vul_scores_list, axis=0)
    benign_scores = np.concatenate(benign_scores_list, axis=0)

    # Calculate training metrics
    pairwise_metrics = calculate_pairwise_metrics(vul_scores, benign_scores, "train")

    result = {"train_loss": float(median_loss)}
    result.update(pairwise_metrics)

    return result



def train(args, train_dataset: PairwiseDataset, model: PairwiseRanker):
    """ Train the model """ 
    
    # ============================
    # Initial Configuration
    # ============================

    # Track gradients
    if args.log_to_wandb:
        wandb.watch(model, pairwise_loss, log="all", log_freq=5)

    # Randomly batch the training dataset 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # Set batch size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)

    # Move the model's parameters and buffers to the specified device.
    model.to(args.device)
    # args.max_steps=args.nb_epochs*len(train_dataloader)
    # args.save_steps=len( train_dataloader)
    args.num_train_epochs=args.nb_epochs

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        
    # Reload parameters when resuming training 
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    # ==================================
    # Variables to follow training stats
    # ==================================

    LOGGER.info("***** Running training *****")
    LOGGER.info("  Num examples = %d", len(train_dataset))
    LOGGER.info("  Num Epochs = %d", args.nb_epochs)
    LOGGER.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    LOGGER.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    best_f1=0.0
    best_pairwise_roc_auc=0.0
    patience = 0
    
    # ============================
    # Actual training
    # ============================
    for epoch in range(args.start_epoch, int(args.num_train_epochs)): 
        
        ### Train model for one epoch
        train_reults = train_one_epoch(model, train_dataloader, optimizer, args, epoch)


        ### Log Model Performance After Every Epoch

        # Log training scores
        for key, value in train_reults.items():
            LOGGER.info("  %s = %s", key, round(value,4))
        if args.log_to_wandb:
            wandb.log(train_reults, step=epoch)

        # Log validation scores
        if args.local_rank in [-1, 0]:
            # save model checkpoint at ep10
            if epoch == 9:
                checkpoint_prefix = f'checkpoint-acsac/'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model-ep{epoch}.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                LOGGER.info(f"ACSAC: Saving model checkpoint at epoch {epoch} to {output_dir}")

            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                validation_results = validate(
                    model, 
                    args.primevul_paired_valid_data_file, 
                    args.primevul_single_input_valid_dataset,
                    args.per_gpu_eval_batch_size,
                    args.local_rank,
                    args.n_gpu,
                    args.device,
                    eval_when_training=True)
                
                for key, value in validation_results.items():
                    LOGGER.info("  %s = %s", key, round(value,4))
                
                # log validation metrics to wandb
                if args.log_to_wandb:
                    wandb.log(validation_results, step=epoch)

                # Save model checkpoint    
                if validation_results['validation_pairwise_roc_auc']>best_pairwise_roc_auc:
                    best_pairwise_roc_auc=validation_results['validation_pairwise_roc_auc']
                    LOGGER.info("  "+"*"*20)  
                    LOGGER.info("  Best Pairwise ROC AUC: %s",round(best_pairwise_roc_auc,4))
                    LOGGER.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-pairwise-roc-auc/'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    LOGGER.info(f"Saving best Pairwise ROC AUC model checkpoint at epoch {epoch} to {output_dir}")
                    patience = 0
                else:
                    patience += 1

                if validation_results['validation_f1']>best_f1:
                    best_f1=validation_results['validation_f1']
                    LOGGER.info("  "+"*"*20)
                    LOGGER.info("  Best f1: %s",round(best_f1,4))
                    LOGGER.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-f1/'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    LOGGER.info(f"Saving best f1 model checkpoint at epoch {epoch} to {output_dir}")
                    patience = 0
                else:
                    patience += 1 

        ### Stop training if model does not improve after multiple epochs
        if patience == args.max_patience:
            LOGGER.info(f"Reached max patience {args.max_patience}. End training now.")
            if best_pairwise_roc_auc == 0.0:
                checkpoint_prefix = f'checkpoint-best-pairwise-roc-auc/'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                LOGGER.info("Saving model checkpoint to %s", output_dir)
            break