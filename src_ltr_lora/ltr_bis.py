#!/usr/bin/env python3
"""
Learning to Rank for Code Vulnerability Detection with Qwen-Coder & LoRA.
Validates every epoch, logs to Weights & Biases, and runs a final test eval in the same run.
"""

import os
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import wandb

# ─── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# send all W&B logs to this project
os.environ["WANDB_PROJECT"] = "ai-vul-detect-pairwise-ltr-lora"


# ─── Data Structures & Loading ─────────────────────────────────────────────────

@dataclass
class CodePair:
    vulnerable: str
    benign: str
    vul_id: str
    vulnerable_hash: str
    benign_hash: str

def load_primevul_pairs(file_path: str) -> List[CodePair]:
    """
    Streams a PrimeVul JSONL file (vuln/benign on alternating lines)
    and returns a list of CodePair objects.
    """
    pairs: List[CodePair] = []
    with open(file_path, "r", encoding="utf-8") as f:
        it = iter(f)
        for vuln_line in it:
            benign_line = next(it, None)
            if not benign_line:
                break
            vuln = json.loads(vuln_line)
            benign = json.loads(benign_line)
            if "func" in vuln and "func" in benign:
                pairs.append(CodePair(
                    vulnerable=vuln["func"],
                    benign=benign["func"],
                    vul_id=str(vuln.get("vul_id", "")),
                    vulnerable_hash=str(vuln.get("func_hash", "")),
                    benign_hash=str(benign.get("func_hash", "")),
                ))
    logging.info(f"Loaded {len(pairs)} pairs from {file_path}")
    return pairs


# ─── Dataset & Collator ────────────────────────────────────────────────────────

class PairwiseCodeDataset(Dataset):
    def __init__(self, pairs: List[CodePair], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        logging.info(f"Dataset ready with {len(pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        pos = self.tokenizer(
            pair.benign,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg = self.tokenizer(
            pair.vulnerable,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "pos_input_ids":      pos.input_ids[0],
            "pos_attention_mask": pos.attention_mask[0],
            "neg_input_ids":      neg.input_ids[0],
            "neg_attention_mask": neg.attention_mask[0],
        }

def create_collate_fn(pad_token_id: int):
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pos_ids  = [x["pos_input_ids"]      for x in batch]
        pos_mask = [x["pos_attention_mask"] for x in batch]
        neg_ids  = [x["neg_input_ids"]      for x in batch]
        neg_mask = [x["neg_attention_mask"] for x in batch]
        return {
            "pos_input_ids":      pad_sequence(pos_ids,  batch_first=True, padding_value=pad_token_id),
            "pos_attention_mask": pad_sequence(pos_mask, batch_first=True, padding_value=0),
            "neg_input_ids":      pad_sequence(neg_ids,  batch_first=True, padding_value=pad_token_id),
            "neg_attention_mask": pad_sequence(neg_mask, batch_first=True, padding_value=0),
        }
    return collate_fn


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_pairwise_metrics(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Runs the model on the dataset, collects positive/negative losses,
    computes Wilcoxon–Mann–Whitney AUC and ROC AUC.
    """
    model.eval().to(device)
    pos_losses, neg_losses = [], []
    for batch in DataLoader(dataset, batch_size=1):
        pos_ids  = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids  = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        with torch.no_grad():
            out_pos = model(input_ids=pos_ids,  attention_mask=pos_mask, labels=pos_ids)
            out_neg = model(input_ids=neg_ids,  attention_mask=neg_mask, labels=neg_ids)
        pos_losses.append(out_pos.loss.item())
        neg_losses.append(out_neg.loss.item())

    u_stat, _ = mannwhitneyu(neg_losses, pos_losses, alternative="greater")
    auc_wmw = u_stat / (len(neg_losses) * len(pos_losses))

    y_true   = [1]*len(neg_losses) + [0]*len(pos_losses)
    y_scores = neg_losses + pos_losses
    auc_roc  = roc_auc_score(y_true, y_scores)

    return {
        "auc_wmw":  auc_wmw,
        "auc_roc":  auc_roc,
    }


# ─── Custom Trainer ────────────────────────────────────────────────────────────

class PairwiseTrainer(Trainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        """
        processing_class replaces the deprecated `tokenizer` kwarg.
        """
        super().__init__(*args, **kwargs)
        self.processor = processing_class

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos_ids  = inputs["pos_input_ids"]
        neg_ids  = inputs["neg_input_ids"]
        pos_mask = inputs["pos_attention_mask"]
        neg_mask = inputs["neg_attention_mask"]

        pos_out = model(input_ids=pos_ids,  attention_mask=pos_mask, labels=pos_ids)
        neg_out = model(input_ids=neg_ids,  attention_mask=neg_mask, labels=neg_ids)

        pos_loss = pos_out.loss
        neg_loss = neg_out.loss

        target = torch.ones_like(pos_loss)
        loss   = torch.nn.MarginRankingLoss(margin=1.0)(neg_loss, pos_loss, target)

        # Log individual losses for more detailed tracking
        if self.state.global_step % self.args.logging_steps == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/pos_loss": pos_loss.mean().item(),
                "train/neg_loss": neg_loss.mean().item(),
                "train/loss_diff": (neg_loss.mean() - pos_loss.mean()).item()
            }, step=self.state.global_step)

        return (loss, pos_out) if return_outputs else loss

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Overrides default evaluate to log under eval/ prefix.

        Args:
            eval_dataset: The dataset to evaluate on.
            **kwargs: Additional arguments to pass to the superclass evaluate method.
        Returns:
            The metrics for the evaluation.
        """
        ds = eval_dataset or self.eval_dataset
        raw = compute_pairwise_metrics(self.model, self.processor, ds, device=self.args.device)

        # Log validation metrics to W&B
        val_metrics = {f"eval/{k}": v for k, v in raw.items()}
        wandb.log(val_metrics, step=self.state.global_step)

        # Also log using trainer's built-in logging for consistency
        metrics = {f"eval_{k}": v for k, v in raw.items()}
        self.log(metrics)
        
        return raw


# ─── Main Training & Evaluation Pipeline ──────────────────────────────────────

def main():
    # file paths
    train_path = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired_16pts.jsonl"
    valid_path = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl"
    test_path  = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"

    # model + tokenizer
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    logging.info(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb_cfg   = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    logging.info("Model & LoRA ready")

    # datasets
    train_ds = PairwiseCodeDataset(load_primevul_pairs(train_path), tokenizer)
    valid_ds = PairwiseCodeDataset(load_primevul_pairs(valid_path), tokenizer)
    test_ds  = PairwiseCodeDataset(load_primevul_pairs(test_path),  tokenizer)
    collate  = create_collate_fn(tokenizer.pad_token_id)

    # training args
    args = TrainingArguments(
        output_dir="qwen_ranker_run",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        fp16=True,

        # ─── LOGGING ─────────────────────────────────
        logging_strategy="steps",       # log every `logging_steps`
        logging_steps=1,
        eval_strategy="epoch",    # run your `evaluate()` once per epoch
        save_strategy="steps",          # or however often you want to checkpoint
        save_steps=500,
        save_total_limit=2,

        remove_unused_columns=False,
        report_to="wandb",
        run_name="train-with-val-and-test",
    )

    # trainer
    trainer = PairwiseTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate,
        processing_class=tokenizer,       # replaces deprecated `tokenizer=...`
    )

    # train + validate
    logging.info("Starting training …")
    trainer.train()
    logging.info("Training completed")

    # final test eval in same run
    logging.info("Running final test evaluation …")
    test_raw = compute_pairwise_metrics(model, tokenizer, test_ds, device=args.device)
    
    # Log test metrics to W&B without train/ prefix
    test_metrics_wandb = {f"test_{k}": v for k, v in test_raw.items()}
    wandb.log(test_metrics_wandb)
    
    logging.info(f"Test metrics: {test_metrics_wandb}")

if __name__ == "__main__":
    main()