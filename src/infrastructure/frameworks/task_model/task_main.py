
from lightning.pytorch.cli import LightningCLI
#from bert_model import BertClassificationModel, MultiTaskBertClassificationModel
from datamodule import TaskDataModule
from task_bert import BertClassificationModel


cli = LightningCLI(BertClassificationModel, TaskDataModule)