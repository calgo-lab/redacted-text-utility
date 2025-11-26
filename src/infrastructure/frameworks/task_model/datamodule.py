import json
import random
from typing import Optional
import csv
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertTokenizerFast
import numpy as np
from datasets import load_dataset
import ast
import pandas as pd
import pickle



class TaskClassificationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        text = [x["text"] for x in data]
        
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [
            torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

          
        labels = torch.stack([x["labels"] for x in data])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }
       
class TaskClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, label_lookup):
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        labels = example["labels"]
        label_ids = [self.label_lookup[x] for x in labels]
        label_idxs = torch.tensor([int(x) for x in label_ids])
        one_hot_labels = torch.zeros(len(self.label_lookup), dtype=torch.float32)
        if len(label_idxs) != 0:
           one_hot_labels[label_idxs] = 1
        return {"text": text, "labels":one_hot_labels}
    

class TaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str = "/Users/toroe/Workspace/redacted-text-utility/data/processed",
        dataset_stem: str = "-00000-of-00001_ne_redacted.parquet",
        redaction_type: str = 'text_redacted_with_semantic_label_mask', #--  'text_redacted_with_random_mask' -- 'text_redacted_with_generic_mask'
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 8,
        max_seq_len: int = 256,
    ):
        super().__init__()
        raw_datasets = load_dataset("parquet", data_files={'train': f"{dataset_path}/train{dataset_stem}", 
                                                       "val": f"{dataset_path}/validation{dataset_stem}", 
                                                       'test': f"{dataset_path}/test{dataset_stem}"})

        if redaction_type != 'None':
            print("Filter for redaction type: ", redaction_type)
            raw_datasets = raw_datasets.map(lambda x: {"text":x[redaction_type]} if x[redaction_type] != None else {"text":x["text"]})
        
        raw_datasets = raw_datasets.select_columns(["text", "intents"])
        raw_datasets = raw_datasets.rename_column("intents", "labels")
        all_labels = sorted({
        label
        for labels in raw_datasets["train"]["labels"]
        if labels
        for label in labels
        })
              
        label_lookup = {v: k for k, v in enumerate(all_labels)}
        
        self.training_data = raw_datasets["train"]
        self.test_data = raw_datasets["test"]
        self.val_data = raw_datasets["val"]

               # build label index
        self.label_lookup = label_lookup
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = TaskClassificationCollator(self.tokenizer, max_seq_len)
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        train = TaskClassificationDataset(
            self.training_data,
            label_lookup=self.label_lookup,
        )


        val = TaskClassificationDataset(
            self.val_data,
            label_lookup=self.label_lookup,
        )

        test = TaskClassificationDataset(
            self.test_data,
            label_lookup=self.label_lookup,
        )
    
        self.train = train
        self.val = val
        self.test = test
        print("Val length: ", len(self.val))
        print("Train Length: ", len(self.train))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
    

