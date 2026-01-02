from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path

from flair.data import Corpus, Dictionary
from flair.datasets import CSVClassificationCorpus
from flair.distributed_utils import launch_distributed
from flair.embeddings import DocumentEmbeddings, TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins.functional.anneal_on_plateau import AnnealingPlugin

from data_handlers.echr_data_handler import EchrDataHandler
from training_scripts.tc.multi_gpu_flair_model_trainer import MultiGpuFlairModelTrainer
from training_scripts.tc.wandb_logger_plugin import WandbLoggerPlugin
from utils.project_utils import ProjectUtils

import csv
import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*")
warnings.filterwarnings("ignore", message=r"No device id is provided via `init_process_group`.*")

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

    if not Path(data_dir_path / "train.csv").exists():
        train_df[['text', 'binary_judgement']].to_csv(data_dir_path / "train.csv", 
                                                      sep='\t', 
                                                      index=False, 
                                                      header=['text', 'label'])
    if not Path(data_dir_path / "dev.csv").exists():
        dev_df[['text', 'binary_judgement']].to_csv(data_dir_path / "dev.csv", 
                                                    sep='\t', 
                                                    index=False, 
                                                    header=['text', 'label'])
    
    if not Path(data_dir_path / "test.csv").exists():
        test_df[['text', 'binary_judgement']].to_csv(data_dir_path / "test.csv", 
                                                     sep='\t', 
                                                     index=False, 
                                                     header=['text', 'label'])
    
    column_name_map = {0: "text", 1: "label"}
    label_type = "label"

    csv.field_size_limit(int(1e8))
    corpus: Corpus = CSVClassificationCorpus(data_dir_path,
                                             column_name_map,
                                             skip_header=True,
                                             delimiter="\t",
                                             label_type=label_type,
                                             train_file='train.csv',
                                             dev_file='dev.csv',
                                             test_file='test.csv')
    
    label_dict: Dictionary = corpus.make_label_dictionary(label_type="label")

    model_dir_path = data_dir_path / f"learning-rate-{learning_rate:.0e}".replace('e-0', 'e-')
    model_dir_path = model_dir_path / f"max-epochs-{max_epochs}"
    model_dir_path = model_dir_path / f"mini-batch-size-{mini_batch_size}"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    document_embeddings: DocumentEmbeddings = TransformerDocumentEmbeddings(
        model=transformer_model_name,
        allow_long_sentences=True,
        cls_pooling="mean",
        fine_tune=True
    )

    classifier = TextClassifier(
        embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type='label',
        multi_label=False
    )

    anneal_plugin = AnnealingPlugin(
        base_path=model_dir_path,
        min_learning_rate=5e-7,
        anneal_factor=0.5,
        patience=2,
        initial_extra_patience=1,
        anneal_with_restarts=True
    )

    models_with_unused_parameters = [
        "google-bert/bert-base-german-cased", 
        "xlm-roberta-large",
        "bert-large-cased"
    ]

    if use_multi_gpu:
        trainer = MultiGpuFlairModelTrainer(
            classifier, 
            corpus, 
            find_unused_parameters=False if transformer_model_name not in models_with_unused_parameters else True
        )
    else:
        trainer = ModelTrainer(classifier, corpus)

    wandb_plugin: WandbLoggerPlugin = None
    if log_to_wandb:
        wandb_plugin = WandbLoggerPlugin(
            entity = wandb_entity, 
            project = project_root.name, 
            name = f"echr-tc__{model_dir_name}__fold-{data_fold_k_value}", 
            config = {
                "transformer_model_name": transformer_model_name, 
                "data_fold": data_fold_k_value, 
                "learning_rate": learning_rate, 
                "max_epochs": max_epochs, 
                "mini_batch_size": mini_batch_size, 
                "sample_size": sample_size
            }, 
            tracked = {
                "train/loss", 
                "dev/loss", 
                "dev/micro avg/precision", 
                "dev/micro avg/recall", 
                "dev/micro avg/f1-score", 
                "dev/macro avg/precision", 
                "dev/macro avg/recall", 
                "dev/macro avg/f1-score", 
                "dev/accuracy"
            }
        )

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    trainer.fine_tune(
        model_dir_path, 
        learning_rate = learning_rate, 
        max_epochs = max_epochs, 
        mini_batch_size = mini_batch_size, 
        eval_batch_size = mini_batch_size, 
        write_weights = True, 
        save_final_model = False, 
        use_final_model_for_eval = False, 
        multi_gpu=use_multi_gpu, 
        use_amp=use_multi_gpu, 
        main_evaluation_metric=("macro avg", "f1-score"), 
        shuffle=False, 
        shuffle_first_epoch=False, 
        attach_default_scheduler=False, 
        plugins = [anneal_plugin, wandb_plugin] if log_to_wandb else [anneal_plugin]
    )

if __name__ == "__main__":

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    if use_multi_gpu:
        launch_distributed(fine_tune)
    else:
        fine_tune()