"""
Learning to Rank for Code Vulnerability Detection.
Uses pairwise learning with Qwen-Coder model and LoRA fine-tuning.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel
)
from peft import LoraConfig, get_peft_model, TaskType
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import wandb

from data_loader import CodePair, load_primevul_pairs, PairwiseCodeDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure wandb project
os.environ["WANDB_PROJECT"] = "ai-vul-detect-pairwise-ltr-lora"



class ModelManager:
    """Handles model initialization and configuration."""
    
    @staticmethod
    def create_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the model and tokenizer with proper configuration.
        
        Args:
            model_name: The name of the model to use.
        Returns:
            A tuple containing the model and tokenizer.
        """
        logging.info(f"Loading tokenizer and model: {model_name}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Initialize base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # Configure and apply LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)
        
        logging.info("Model and LoRA configuration loaded")
        return model, tokenizer


class PairwiseTrainer(Trainer):
    """Custom trainer for pairwise learning with margin ranking loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Unpack inputs
        pos_ids = inputs["pos_input_ids"]
        neg_ids = inputs["neg_input_ids"]
        pos_attn = inputs["pos_attention_mask"]
        neg_attn = inputs["neg_attention_mask"]

        # Forward pass
        pos_outputs = model(input_ids=pos_ids, attention_mask=pos_attn, labels=pos_ids)
        neg_outputs = model(input_ids=neg_ids, attention_mask=neg_attn, labels=neg_ids)

        # Compute losses
        pos_loss = pos_outputs.loss
        neg_loss = neg_outputs.loss

        # Margin ranking loss: want neg_loss > pos_loss by margin
        target = torch.ones_like(pos_loss)
        loss = torch.nn.MarginRankingLoss(margin=1.0)(neg_loss, pos_loss, target)

        return (loss, pos_outputs) if return_outputs else loss


class Evaluator:
    """Handles model evaluation and metrics computation."""
    
    @staticmethod
    def evaluate_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_path: str,
        device: str = "cuda"
    ) -> Tuple[float, float]:
        """Evaluate model performance using AUC metrics."""
        code_pairs = load_primevul_pairs(test_path)
        model.to(device)
        model.eval()
        
        pos_losses, neg_losses = [], []
        for pair in code_pairs:
            # Tokenize
            enc_pos = tokenizer(
                pair.benign,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=1024
            ).to(device)
            
            enc_neg = tokenizer(
                pair.vulnerable,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=1024
            ).to(device)

            # Compute losses
            with torch.no_grad():
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
        u_stat, _ = mannwhitneyu(neg_losses, pos_losses, alternative="greater")
        auc_wmw = u_stat / (len(neg_losses) * len(pos_losses))
        
        y_true = [1] * len(neg_losses) + [0] * len(pos_losses)
        y_scores = neg_losses + pos_losses
        auc_sklearn = roc_auc_score(y_true, y_scores)

        # Log metrics
        wandb.log({
            "test_auc_wmw": auc_wmw,
            "test_auc_roc": auc_sklearn
        })
        
        logging.info(f"WMW AUC: {auc_wmw:.4f}, ROC AUC: {auc_sklearn:.4f}")
        return auc_wmw, auc_sklearn


def create_collate_fn(pad_token_id: int):
    """Creates a collate function for batching."""
    
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch received")
            
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
    
    return collate_fn

def main():
    """Main training and evaluation pipeline."""
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    model, tokenizer = ModelManager.create_model_and_tokenizer(model_name)
    
    # Load training data
    train_path = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
    train_pairs = load_primevul_pairs(train_path)
    
    # Create dataset and collate function
    train_dataset = PairwiseCodeDataset(train_pairs, tokenizer)
    collate_fn = create_collate_fn(tokenizer.pad_token_id)

    # Configure training
    training_args = TrainingArguments(
        output_dir="qwen_ranker_test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        fp16=True,
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="test_5",
    )

    # Initialize and run trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed")

    # Evaluate model
    logging.info("Evaluating model...")
    test_path = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl"
    Evaluator.evaluate_model(model, tokenizer, test_path)
    logging.info("Evaluation completed")

if __name__ == "__main__":
    main()
