import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    EvalPrediction,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import wandb

from data_loader import CodePair, load_primevul_pairs, PairwiseCodeDataset
import argparse


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    train_data_path: str = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
    eval_data_path: str = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl"
    test_data_path: str = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl"
    
    # Model configuration
    max_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    
    # Training configuration
    output_dir: str = "qwen_ranker_checkpoints"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 5
    margin: float = 1.0
    
    # Logging and saving
    logging_steps: int = 1
    eval_steps: int = 5
    save_steps: int = 10
    save_total_limit: int = 2
    wandb_run_name: str = "ltr_Qwen2.5-Coder-32B_r16_a32_lr1e-4_batch8_epoch5_16pts"
    wandb_project: str = "ai-vul-detect-pairwise-ltr-lora"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class Logger:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup_logging(level: int = logging.INFO) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        return logging.getLogger(__name__)


class ModelManager:
    """Handles model initialization and configuration."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the model and tokenizer with proper configuration."""
        self.logger.info(f"Loading tokenizer and model: {self.config.model_name}")
        
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Initialize base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Configure and apply LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
            
            self.logger.info("Model and LoRA configuration loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise


class DataManager:
    """Handles data loading and dataset creation."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_datasets(self, tokenizer: PreTrainedTokenizer) -> Tuple[PairwiseCodeDataset, PairwiseCodeDataset, List[CodePair]]:
        """Load training, evaluation, and test datasets."""
        self.logger.info("Loading datasets...")
        
        try:
            # Load data pairs
            train_pairs = load_primevul_pairs(self.config.train_data_path)
            eval_pairs = load_primevul_pairs(self.config.eval_data_path)
            test_pairs = load_primevul_pairs(self.config.test_data_path)
            
            # Create datasets
            train_dataset = PairwiseCodeDataset(train_pairs, tokenizer, self.config.max_length)
            eval_dataset = PairwiseCodeDataset(eval_pairs, tokenizer, self.config.max_length)
            
            self.logger.info(f"Loaded {len(train_pairs)} training pairs, "
                           f"{len(eval_pairs)} evaluation pairs, "
                           f"{len(test_pairs)} test pairs")
            
            return train_dataset, eval_dataset, test_pairs
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise


class PairwiseTrainer(Trainer):
    """Custom trainer for pairwise learning with margin ranking loss."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.logger = logger
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute pairwise margin ranking loss."""
        try:
            # Unpack inputs
            pos_ids = inputs["pos_input_ids"]
            neg_ids = inputs["neg_input_ids"]
            pos_attn = inputs["pos_attention_mask"]
            neg_attn = inputs["neg_attention_mask"]

            # Forward pass
            pos_outputs = model(input_ids=pos_ids, attention_mask=pos_attn, labels=pos_ids)
            neg_outputs = model(input_ids=neg_ids, attention_mask=neg_attn, labels=neg_ids)

            # Extract losses
            pos_loss = pos_outputs.loss
            neg_loss = neg_outputs.loss

            # Margin ranking loss: we want neg_loss > pos_loss by margin
            # This encourages the model to assign higher loss to vulnerable code
            target = torch.ones_like(pos_loss)
            margin_loss = torch.nn.MarginRankingLoss(margin=self.config.margin)
            loss = margin_loss(neg_loss, pos_loss, target)

            return (loss, pos_outputs) if return_outputs else loss
            
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            raise


class Evaluator:
    """Handles model evaluation and metrics computation."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def evaluate_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        code_pairs: List[CodePair],
        dataset_name: str = "validation"
    ) -> Dict[str, float]:
        """Evaluate model performance using AUC metrics."""
        self.logger.info(f"Evaluating model on {dataset_name} set...")
        
        try:
            device = next(model.parameters()).device
            model.eval()
            
            pos_losses, neg_losses = [], []
            
            with torch.no_grad():
                for pair in code_pairs:
                    # Tokenize benign code
                    enc_pos = tokenizer(
                        pair.benign,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=self.config.max_length
                    ).to(device)
                    
                    # Tokenize vulnerable code
                    enc_neg = tokenizer(
                        pair.vulnerable,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=self.config.max_length
                    ).to(device)

                    # Compute losses
                    pos_out = model(
                        input_ids=enc_pos.input_ids,
                        attention_mask=enc_pos.attention_mask,
                        labels=enc_pos.input_ids
                    )
                    neg_out = model(
                        input_ids=enc_neg.input_ids,
                        attention_mask=enc_neg.attention_mask,
                        labels=enc_neg.input_ids
                    )
                    
                    pos_losses.append(pos_out.loss.item())
                    neg_losses.append(neg_out.loss.item())
            
            # Compute metrics
            metrics = self._compute_metrics(pos_losses, neg_losses, dataset_name)
            
            self.logger.info(f"{dataset_name} results - WMW AUC: {metrics[f'{dataset_name}_auc_wmw']:.4f}, "
                           f"ROC AUC: {metrics[f'{dataset_name}_auc_roc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _compute_metrics(self, pos_losses: List[float], neg_losses: List[float], prefix: str) -> Dict[str, float]:
        """Compute AUC metrics from losses."""
        # Mann-Whitney U test AUC
        u_stat, _ = mannwhitneyu(neg_losses, pos_losses, alternative="greater")
        auc_wmw = u_stat / (len(neg_losses) * len(pos_losses))
        
        # ROC AUC
        y_true = [1] * len(neg_losses) + [0] * len(pos_losses)
        y_scores = neg_losses + pos_losses
        auc_sklearn = roc_auc_score(y_true, y_scores)
        
        return {
            f"{prefix}_auc_wmw": auc_wmw,
            f"{prefix}_auc_roc": auc_sklearn
        }


def create_collate_fn(pad_token_id: int, logger: logging.Logger):
    """Creates a collate function for batching."""
    
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for pairwise data."""
        if not batch:
            logger.error("Empty batch received")
            raise ValueError("Empty batch received")
        
        try:
            # Collect tensors
            pos_input_ids = [x["pos_input_ids"] for x in batch]
            pos_attention = [x["pos_attention_mask"] for x in batch]
            neg_input_ids = [x["neg_input_ids"] for x in batch]
            neg_attention = [x["neg_attention_mask"] for x in batch]
            
            # Pad and stack
            return {
                "pos_input_ids": pad_sequence(pos_input_ids, batch_first=True, padding_value=pad_token_id),
                "pos_attention_mask": pad_sequence(pos_attention, batch_first=True, padding_value=0),
                "neg_input_ids": pad_sequence(neg_input_ids, batch_first=True, padding_value=pad_token_id),
                "neg_attention_mask": pad_sequence(neg_attention, batch_first=True, padding_value=0),
            }
        except Exception as e:
            logger.error(f"Error in collate function: {str(e)}")
            raise
    
    return collate_fn


class TrainingPipeline:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = Logger.setup_logging()
        
        # Initialize components
        self.model_manager = ModelManager(config, self.logger)
        self.data_manager = DataManager(config, self.logger)
        self.evaluator = Evaluator(config, self.logger)
        
        # Setup wandb
        os.environ["WANDB_PROJECT"] = config.wandb_project
    
    def run(self):
        """Execute the complete training pipeline."""
        try:
            self.logger.info("Starting training pipeline...")
            
            # Initialize model and tokenizer
            model, tokenizer = self.model_manager.create_model_and_tokenizer()
            
            # Load datasets
            train_dataset, eval_dataset, test_pairs = self.data_manager.load_datasets(tokenizer)
            
            # Create collate function
            collate_fn = create_collate_fn(tokenizer.pad_token_id, self.logger)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.num_train_epochs,
                fp16=True,
                logging_steps=self.config.logging_steps,
                eval_steps=self.config.eval_steps,
                eval_strategy="no",  # Disable built-in evaluation to avoid conflicts
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=False,
                remove_unused_columns=False,
                report_to="wandb",
                run_name=self.config.wandb_run_name,
                dataloader_pin_memory=False,
                warmup_steps=100,
            )
            
            # Custom evaluation function for trainer
            def compute_metrics_for_trainer(eval_pred: EvalPrediction) -> Dict[str, float]:
                """Custom metrics computation for trainer's built-in evaluation."""
                # Return empty dict to avoid issues with built-in evaluation
                # Our custom evaluation happens in the callback
                return {}
            
            # Initialize trainer with disabled built-in evaluation
            trainer = PairwiseTrainer(
                config=self.config,
                logger=self.logger,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                # Remove eval_dataset to avoid input_ids conflict
                data_collator=collate_fn,
            )
            
            # Custom callback for step-based evaluation
            class StepBasedEvaluationCallback(TrainerCallback):
                def __init__(self, evaluator, tokenizer, eval_data_path, logger, eval_steps):
                    self.evaluator = evaluator
                    self.tokenizer = tokenizer
                    self.eval_data_path = eval_data_path
                    self.logger = logger
                    self.eval_steps = eval_steps
                    self.eval_pairs_data = None
                
                def on_step_end(self, args, state, control, model=None, **kwargs):
                    """Called after each training step to check if evaluation should run."""
                    # Check if we should evaluate at this step
                    if state.global_step % self.eval_steps == 0:
                        try:
                            # Load eval data if not already loaded
                            if self.eval_pairs_data is None:
                                self.eval_pairs_data = load_primevul_pairs(self.eval_data_path)
                            
                            # Run custom evaluation
                            eval_metrics = self.evaluator.evaluate_model(
                                model, self.tokenizer, self.eval_pairs_data, "validation"
                            )
                            
                            # Log to wandb
                            wandb.log({
                                **eval_metrics, 
                                "step": state.global_step,
                                "epoch": state.epoch
                            })
                            
                            self.logger.info(f"Step {state.global_step} evaluation completed")
                            
                        except Exception as e:
                            self.logger.error(f"Error during step-based evaluation: {str(e)}")
            
            # Add the step-based evaluation callback
            callback = StepBasedEvaluationCallback(
                self.evaluator, tokenizer, self.config.eval_data_path, self.logger, self.config.eval_steps
            )
            trainer.add_callback(callback)
            
            # Start training
            self.logger.info("Starting training...")
            trainer.train()
            self.logger.info("Training completed successfully")
            
            # Final evaluation on test set
            self.logger.info("Evaluating on test set...")
            test_metrics = self.evaluator.evaluate_model(model, tokenizer, test_pairs, "test")
            wandb.log(test_metrics)
            
            # Save final model
            final_model_path = Path(self.config.output_dir) / "final_model"
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")
            
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train pairwise LTR model for vulnerability detection")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-Coder-32B")
    parser.add_argument('--train_data_path', type=str, default="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl")
    parser.add_argument('--eval_data_path', type=str, default="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl")
    parser.add_argument('--test_data_path', type=str, default="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default="qwen_ranker_checkpoints")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--wandb_run_name', type=str, default="ltr_Qwen2.5-Coder-32B_r16_a32_lr1e-4_batch8_epoch5_16pts")
    parser.add_argument('--wandb_project', type=str, default="ai-vul-detect-pairwise-ltr-lora")

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        test_data_path=args.test_data_path,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        margin=args.margin,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        wandb_run_name=args.wandb_run_name,
        wandb_project=args.wandb_project
    )

    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    print(f"Training completed! Final test results: {results}")


if __name__ == "__main__":
    main()