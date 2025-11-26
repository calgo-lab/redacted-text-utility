from typing import Optional, Dict, Any, List, Union
import os
import lightning.pytorch as pl
import torch
import torchmetrics
import pandas as pd
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelRecall, MultilabelPrecision
import transformers
from transformers import  AutoModel
import torch.nn.functional as F
from torchmetrics.functional import f1_score, precision, recall, accuracy, auroc, average_precision
from pathlib import Path


class BertClassificationModel(pl.LightningModule):
    def __init__(self,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 num_classes: int = 20,
                 warmup_steps: int = 500,
                 decay_steps: int = 50_000,
                 num_training_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 device: str = "cuda",
                 save_scores: bool = False,
                 hidden_dim=768,
                 eval_treshold: float = 0.25,
                 test_out: str = ""
                 ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.num_classes = num_classes
        self.classification_layers = torch.nn.Linear(hidden_dim, self.num_classes)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr

        self.val_output = []
        self.test_output = []
        self.save_scores = save_scores
        self.eval_treshold = eval_treshold
        self.test_out = test_out
    def forward(self,
                batch):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        task_probs = {}
        logits = self.classification_layers(encoded)
        return torch.sigmoid(logits)
              

    def training_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layers(encoded)
        labels = batch["labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("Train/Loss", loss)
        return loss


    def test_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]

        logits = self.classification_layers(encoded) 
        self.test_output.append({"logits": logits, "labels": batch["labels"]})
        return {"logits": logits, "labels": batch["labels"]}

    def on_test_epoch_end(self) -> None:

        logits = torch.cat([x["logits"] for x in self.test_output])
        labels = torch.cat([x["labels"] for x in self.test_output]).int() 
        preds = torch.sigmoid(logits)
        
        macro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")
        micro_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="micro")
        all_score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="none")

        macro_score_recall = recall(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")
        micro_score_recall = recall(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="micro")
        all_score_recall = recall(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="none")

        macro_score_precision = precision(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")
        micro_score_precision = precision(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="micro")
        all_score_precision = precision(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="none")

        macro_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")
        micro_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="micro")   
        all_score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="none")           
        
        macro_score_auroc = auroc(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        micro_score_auroc = auroc(preds, labels, 'multilabel', num_labels=self.num_classes, average="micro")   
        all_score_auroc = auroc(preds, labels, 'multilabel', num_labels=self.num_classes, average="none")

        macro_score_ap = average_precision(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        micro_score_ap = average_precision(preds, labels, 'multilabel', num_labels=self.num_classes, average="micro")   
        all_score_ap = average_precision(preds, labels, 'multilabel', num_labels=self.num_classes, average="none")

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        
        self.log(f"Test/F1_macro", macro_score_f1, sync_dist=True)
        self.log(f"Test/Precision_macro", macro_score_precision, sync_dist=True)
        self.log(f"Test/Recall_macro", macro_score_recall, sync_dist=True)
        self.log(f"Test/Accuracy_macro", macro_score_accuracy, sync_dist=True)
        self.log(f"Test/Auroc_macro", macro_score_auroc, sync_dist=True)
        self.log(f"Test/AP_macro", macro_score_ap, sync_dist=True)
        self.log(f"Test/Loss", loss, sync_dist=True)
        
        results = {"f1": all_score_f1.tolist(),\
                    "recall": all_score_recall.tolist(),\
                    "precision":all_score_precision.tolist(),\
                    "accuracy": all_score_accuracy.tolist(),\
                    "auroc": all_score_auroc.tolist(),\
                    "accuracy": all_score_ap.tolist()}
        

        if self.save_scores:
            experiment_df = pd.DataFrame.from_dict(results)
            Path(self.test_out).mkdir(exist_ok=True)
            experiment_df.to_json(f"{self.test_out}/results.json")
           
        self.test_output = list()


    def validation_step(self, batch, batch_idx,**kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layers(encoded)
        self.val_output.append({"logits": logits, "labels": batch["labels"]})
        return {"logits": logits, "labels": batch["labels"]}
        
    
    def on_validation_epoch_end(self) -> None:


        logits = torch.cat([x["logits"] for x in self.val_output])
        labels = torch.cat([x["labels"] for x in self.val_output]).int() 
        preds = torch.sigmoid(logits)
        score_f1 = f1_score(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")\
                
        score_recall = recall(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")\
                
        score_precision = precision(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")\
                
        score_accuracy = accuracy(preds, labels, 'multilabel', num_labels=self.num_classes, threshold=self.eval_treshold, average="macro")\
        
        score_auroc = auroc(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
        
        score_ap = average_precision(preds, labels, 'multilabel', num_labels=self.num_classes, average="macro")
                
        
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
 
        
        self.log(f"Val/F1", score_f1, sync_dist=True)
        self.log(f"Val/Precision", score_precision, sync_dist=True)
        self.log(f"Val/Recall", score_recall, sync_dist=True)
        self.log(f"Val/Accuracy", score_accuracy, sync_dist=True)
        self.log(f"Val/Auroc", score_auroc, sync_dist=True)
        self.log(f"Val/AP", score_ap, sync_dist=True)
        self.log(f"Val/Loss", loss, sync_dist=True)        
        self.log("Val/Weighted_loss", loss)
        self.val_output = list()
        
    
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
    def eval(self):
        self.encoder.eval()
        self.classification_layers.eval()
           
