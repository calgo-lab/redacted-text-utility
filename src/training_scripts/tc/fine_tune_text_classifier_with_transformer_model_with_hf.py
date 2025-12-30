from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

from data_handlers.echr_data_handler import EchrDataHandler
from utils.project_utils import ProjectUtils

import os
import random

import numpy as np
import pandas as pd
import torch
import wandb


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_csv_dataset(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", sep="\t")
    dev = pd.read_csv(data_dir / "dev.csv", sep="\t")
    test = pd.read_csv(data_dir / "test.csv", sep="\t")

    return DatasetDict({
        "train": Dataset.from_pandas(train),
        "validation": Dataset.from_pandas(dev),
        "test": Dataset.from_pandas(test),
    })

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "micro/precision": p_micro,
        "micro/recall": r_micro,
        "micro/f1": f1_micro,
        "macro/precision": p_macro,
        "macro/recall": r_macro,
        "macro/f1": f1_macro
    }


def fine_tune():

    model_checkpoints_root_dir = os.environ.get("MODEL_CHECKPOINTS_ROOT_DIR", None)
    model_checkpoints_root_dir = Path(model_checkpoints_root_dir) if model_checkpoints_root_dir else Path.home() / "model_checkpoints"

    data_dir = os.environ.get("DATA_DIR", None)
    data_dir = Path(data_dir) if data_dir else None

    data_fold_k_value = os.environ.get("DATA_FOLD_K_VALUE", None)
    data_fold_k_value = int(data_fold_k_value) if data_fold_k_value else 1

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    log_to_wandb = os.environ.get("LOG_TO_WANDB", None)
    log_to_wandb = int(log_to_wandb) if log_to_wandb else 0
    log_to_wandb = bool(log_to_wandb) if log_to_wandb else False

    if log_to_wandb:
        wandb_entity = os.environ.get("WANDB_ENTITY", "sksdotsauravs-dev")

    transformer_model_name = os.environ.get("TRANSFORMER_MODEL_NAME", "google-bert/bert-base-german-cased")

    learning_rate = os.environ.get("LEARNING_RATE", None)
    learning_rate = float(learning_rate) if learning_rate else 5e-5
    
    max_epochs = os.environ.get("MAX_EPOCHS", None)
    max_epochs = int(max_epochs) if max_epochs else 35
    
    mini_batch_size = os.environ.get("MINI_BATCH_SIZE", None)
    mini_batch_size = int(mini_batch_size) if mini_batch_size else 4
    
    project_root: Path = ProjectUtils.get_project_root()
    data_handler = EchrDataHandler(project_root, data_dir=data_dir)
    datasetdict = data_handler.get_train_dev_test_datasetdict(k=data_fold_k_value)

    train_df = datasetdict["train"].to_pandas()
    dev_df = datasetdict["dev"].to_pandas()
    test_df = datasetdict["test"].to_pandas()
    sample_size = len(train_df) + len(dev_df) + len(test_df)

    print(f"model_checkpoints_root_dir: {model_checkpoints_root_dir}")
    print(f"data_dir: {data_dir}")
    print(f"data_fold_k_value: {data_fold_k_value}")
    print(f"use_multi_gpu: {use_multi_gpu}")
    print(f"log_to_wandb: {log_to_wandb}")
    if log_to_wandb:
        print(f"wandb_entity: {wandb_entity}")
    print(f"transformer_model_name: {transformer_model_name}")
    print(f"learning_rate: {learning_rate:.0e}".replace('e-0', 'e-'))
    print(f"max_epochs: {max_epochs}")
    print(f"mini_batch_size: {mini_batch_size}")
    print(f"sample_size: {sample_size}")

    model_dir_name = transformer_model_name.replace("/", "--").replace("_", "-")

    data_dir_path = model_checkpoints_root_dir / "echr" / "tc" / model_dir_name
    data_dir_path = data_dir_path / "additional-embeddings-none"

    data_dir_path =  data_dir_path / f"sample-size-{sample_size}" / f"data-fold-{data_fold_k_value}"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    train_df[['text', 'binary_judgement']].to_csv(data_dir_path / "train.csv", 
                                                  sep='\t', 
                                                  index=False, 
                                                  header=['text', 'label'])
    
    dev_df[['text', 'binary_judgement']].to_csv(data_dir_path / "dev.csv", 
                                                sep='\t', 
                                                index=False, 
                                                header=['text', 'label'])
    
    test_df[['text', 'binary_judgement']].to_csv(data_dir_path / "test.csv", 
                                                 sep='\t', 
                                                 index=False, 
                                                 header=['text', 'label'])
    
    dataset = load_csv_dataset(data_dir_path)

    set_seed(2025)

    tokenizer = AutoTokenizer.from_pretrained(
        transformer_model_name,
        model_max_length=4096,
        padding=False,
        truncation=True
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer.model_max_length
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    num_labels = len(set(dataset["train"]["labels"]))

    model = AutoModelForSequenceClassification.from_pretrained(
        transformer_model_name,
        num_labels=num_labels,
        gradient_checkpointing=True
    )

    if log_to_wandb:
        wandb.init(
            entity=wandb_entity,
            project="echr-text-classification",
            name=f"{transformer_model_name.replace('/', '_')}",
            config={
                "transformer_model_name": transformer_model_name,
                "data_fold": data_fold_k_value, 
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "mini_batch_size": mini_batch_size,
                "sample_size": sample_size
            }
        )
    
    model_dir_path = data_dir_path / f"learning-rate-{learning_rate:.0e}".replace('e-0', 'e-')
    model_dir_path = model_dir_path / f"max-epochs-{max_epochs}"
    model_dir_path = model_dir_path / f"mini-batch-size-{mini_batch_size}"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_dir_path),
        learning_rate=learning_rate,
        num_train_epochs=max_epochs,

        per_device_train_batch_size=mini_batch_size,
        per_device_eval_batch_size=mini_batch_size,
        gradient_accumulation_steps=mini_batch_size,

        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="macro/f1",
        greater_is_better=True,

        save_total_limit=2,

        report_to="wandb" if log_to_wandb else "none",
        run_name=f"{transformer_model_name.replace('/', '_')}",

        resume_from_checkpoint=True
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = trainer.evaluate(dataset["test"])
    print(metrics)

    if log_to_wandb:
        wandb.finish()

if __name__ == "__main__":
    fine_tune()