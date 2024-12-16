"""All functions related to training the model
"""

import os
import torch

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

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from src.utils import LOGGER
from src.train_eval import validate, pairwise_loss

def train(args, train_dataset, model):
    """ Train the model """ 
    
    # ============================
    # Initial Configuration
    # ============================

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

    global_step = args.start_step
    avg_loss = 0.0
    best_f1=0.0
    best_acc=0.0
    patience = 0

    # To log training stats
    short_comment_str = args.output_dir.split('/')[-1]
    tensorboard_logdir = f'{args.run_dir}/{short_comment_str}'
    # TODO DELETE
    # writer = SummaryWriter(tensorboard_logdir)
    
    # ============================
    # Actual training
    # ============================
    for epoch in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        cumul_train_loss=0 # Tracks cumulative loss
        for local_step, batch in enumerate(bar):
            code_vulnerable = batch["vulnerable_code"]
            code_benign = batch["benign_code"]
            
            ### Forward pass and loss
            score_vuln, score_benign = model(code_vulnerable, code_benign)
            
            loss = pairwise_loss(score_vuln, score_benign)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            ### Gradient descent
            optimizer.zero_grad() # Clear gradients for next train
            loss.backward() # Backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # Prevent exploding gradients
            optimizer.step() # Apply gradients

            # Log gradients stats
            ######################################################################################################################################
            # log_gradients(model, epoch, global_step, LOGGER)

            ### Update training stats and move to prepare moving to the next iteration
            cumul_train_loss+=loss.item()
            avg_loss = round(cumul_train_loss/(local_step + 1),5) #if avg_loss!=0 else tr_loss
            bar.set_description(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss}")

            # Tracks the exponential rate of change in the training loss over a specific number of steps. 
            # Useful to monitor a smoothed loss trend in training processes.
            # avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step + 1 - tr_nb)),4)
            # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     logging_loss = tr_loss
            #     tr_nb=global_step


            ### Log after every logging_steps and save model if better performance
            if (global_step + 1) % args.logging_steps == 0 and args.local_rank == -1 and args.evaluate_during_training: # Only evaluate when single GPU otherwise metrics may not average well

                ### Evaluate model
                # Valid perf
                results = validate(
                    model, 
                    args.primevul_paired_valid_data_file, 
                    args.primevul_single_input_valid_dataset,
                    args.eval_batch_size,
                    args.local_rank,
                    args.n_gpu,
                    args.device,
                    eval_when_training=True)
                
                for key, value in results.items():
                    LOGGER.info("  %s = %s", key, round(value,4))

                # Save model checkpoint    
                if results['eval_f1']>best_f1:
                    best_f1=results['eval_f1']
                    LOGGER.info("  "+"*"*20)  
                    LOGGER.info("  Best f1:%s",round(best_f1,4))
                    LOGGER.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-f1/'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    LOGGER.info(f"Saving best f1 model checkpoint at epoch {epoch} step {global_step} to {output_dir}")   

            # increment global step
            global_step += 1
        
        ### log after every epoch
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
                results = validate(
                    model, 
                    args.primevul_paired_valid_data_file, 
                    args.primevul_single_input_valid_dataset,
                    args.eval_batch_size,
                    args.local_rank,
                    args.n_gpu,
                    args.device,
                    eval_when_training=True)
                
                for key, value in results.items():
                    LOGGER.info("  %s = %s", key, round(value,4))
                
                # Save model checkpoint    
                if results['eval_f1']>best_f1:
                    best_f1=results['eval_f1']
                    LOGGER.info("  "+"*"*20)  
                    LOGGER.info("  Best f1:%s",round(best_f1,4))
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

                if results['eval_acc']>best_acc:
                    best_acc=results['eval_acc']
                    LOGGER.info("  "+"*"*20)
                    LOGGER.info("  Best acc:%s",round(best_acc,4))
                    LOGGER.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-acc/'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    LOGGER.info(f"Saving best acc model checkpoint at epoch {epoch} to {output_dir}")
                    patience = 0
                else:
                    patience += 1 

        ### Stop training if model does not improve after multiple epochs
        if patience == args.max_patience:
            LOGGER.info(f"Reached max patience {args.max_patience}. End training now.")
            if best_f1 == 0.0:
                checkpoint_prefix = f'checkpoint-best-f1/'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                LOGGER.info("Saving model checkpoint to %s", output_dir)
            break

    # writer.close()

def log_gradients(model, epoch, step, logger=None):
    """
    Logs the gradients of model parameters, including L2 norms.
    If `logger` is provided, logs will be sent there. Otherwise, prints to stdout.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.max().item()
            grad_min = param.grad.min().item()
            grad_norm = param.grad.norm(2).item()  # L2 norm of gradients
            message = (f"Epoch {epoch}, Step {step}, Param: {name}, "
                       f"Grad Mean: {grad_mean:.4e}, Grad Std: {grad_std:.4e}, "
                       f"Grad Max: {grad_max:.4e}, Grad Min: {grad_min:.4e}, "
                       f"Grad Norm (L2): {grad_norm:.4e}")
            if logger:
                logger.info(message)
            else:
                print(message)