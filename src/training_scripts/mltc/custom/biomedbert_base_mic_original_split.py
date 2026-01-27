from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path

from flair.data import Corpus, Dictionary
from flair.datasets import ClassificationCorpus
from flair.distributed_utils import launch_distributed
from flair.embeddings import DocumentEmbeddings, TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from transformers import AutoTokenizer

from tqdm import tqdm

from data_handlers.mic_data_handler import MicDataHandler
from training_scripts.tc.multi_gpu_flair_model_trainer import MultiGpuFlairModelTrainer
from training_scripts.tc.wandb_logger_plugin import WandbLoggerPlugin
from utils.project_utils import ProjectUtils

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*")
warnings.filterwarnings("ignore", message=r"No device id is provided via `init_process_group`.*")

def convert_to_fasttext_format(row: pd.Series) -> str:
    """
    Converts a row with 'text' and 'intents' columns to FastText format.
    
    :param row: A pandas Series with 'text' and 'intents' columns.
    :return: A string in FastText format: "__label__<label1> __label__<label2> ... <text>"
    """

    labels = " ".join([f"__label__{label.replace(' ', '_')}" for label in row['intents']])
    text = row['text'].replace("\n", " ").replace("\t", " ")
    return f"{labels} {text}"

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
    data_handler = MicDataHandler(project_root, data_dir=data_dir)

    train_df = data_handler.get_dataframe_for_file("train-00000-of-00001.parquet")
    dev_df = data_handler.get_dataframe_for_file("validation-00000-of-00001.parquet")
    test_df = data_handler.get_dataframe_for_file("test-00000-of-00001.parquet")
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

    data_dir_path = model_checkpoints_root_dir / "mic" / "mltc" / model_dir_name
    data_dir_path = data_dir_path / "additional-embeddings-none"

    data_dir_path =  data_dir_path / f"sample-size-{sample_size}" / f"data-fold-{data_fold_k_value}"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    MAX_TOKENS = 510 # leave 2 tokens for special tokens, e.g., [CLS], [SEP]

    def chunk_text_to_fasttext(text: str, labels: list[str]) -> list[str]:
        """
        Splits a long text into chunks of MAX_TOKENS, prepends the labels
        in FastText format, returns list of lines for ClassificationCorpus.

        :param text: The input text to be chunked.
        :param labels: List of labels associated with the text.
        :return: List of strings, each in FastText format.
        """
        label_str = " ".join([f"__label__{label.replace(' ', '_')}" for label in labels])
        
        tokens = tokenizer.tokenize(text)

        chunks = list()
        for i in range(0, len(tokens), MAX_TOKENS):
            chunk_tokens = tokens[i: i+MAX_TOKENS]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(f"{label_str} {chunk_text}")
        return chunks

    def write_chunked_fasttext_file(df, path: Path):
        """
        Writes a DataFrame to a FastText format file with chunked text.

        :param df: DataFrame with 'text' and 'intents' columns.
        :param path: Path to the output file.
        """
        with open(path, "w", encoding="utf-8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                text_chunks = chunk_text_to_fasttext(row['text'], row['intents'])
                for chunk in text_chunks:
                    f.write(chunk + "\n")

    write_chunked_fasttext_file(train_df, data_dir_path / "train.txt")
    write_chunked_fasttext_file(dev_df, data_dir_path / "dev.txt")
    write_chunked_fasttext_file(test_df, data_dir_path / "test.txt")

    corpus: Corpus = ClassificationCorpus(data_dir_path,
                                          train_file='train.txt',
                                          dev_file='dev.txt',
                                          test_file='test.txt',
                                          label_type='intents')
    
    label_dict: Dictionary = corpus.make_label_dictionary(label_type="intents")

    model_dir_path = data_dir_path / f"learning-rate-{learning_rate:.0e}".replace('e-0', 'e-')
    model_dir_path = model_dir_path / f"max-epochs-{max_epochs}"
    model_dir_path = model_dir_path / f"mini-batch-size-{mini_batch_size}"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    document_embeddings: DocumentEmbeddings = TransformerDocumentEmbeddings(
        model=transformer_model_name,
        fine_tune=True,
        transformers_tokenizer_kwargs={
            "truncation": True,
            "max_length": 512,
            "padding": "max_length"
        }
    )

    classifier = TextClassifier(
        embeddings=document_embeddings,
        label_dictionary=label_dict,
        label_type='intents',
        multi_label=True
    )

    models_with_unused_parameters = [
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
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
            name = f"mic-mltc__{model_dir_name}__fold-{data_fold_k_value}", 
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

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    
    trainer.fine_tune(
        model_dir_path, 
        learning_rate = learning_rate, 
        max_epochs = max_epochs, 
        mini_batch_size = mini_batch_size, 
        eval_batch_size = mini_batch_size, 
        main_evaluation_metric=("macro avg", "f1-score"),
        write_weights = True, 
        save_final_model = False, 
        use_final_model_for_eval = False, 
        multi_gpu = use_multi_gpu, 
        use_amp = use_multi_gpu, 
        shuffle = False if use_multi_gpu else True, 
        shuffle_first_epoch = False if use_multi_gpu else True, 
        plugins = [wandb_plugin] if log_to_wandb else None
    )

if __name__ == "__main__":

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    if use_multi_gpu:
        launch_distributed(fine_tune)
    else:
        fine_tune()