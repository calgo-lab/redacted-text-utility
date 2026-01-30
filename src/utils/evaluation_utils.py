from pathlib import Path
from typing import List

from flair.data import Corpus
from flair.datasets import ClassificationCorpus, CSVClassificationCorpus
from flair.models import TextClassifier
from flair.training_utils import Result
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.token_treatment_utils import TokenTreatmentUtils

import csv

import pandas as pd


class EvaluationUtils:

    @staticmethod
    def redact_and_evaluate_for_text_classifier(input_df: pd.DataFrame,
                                                pe_df: pd.DataFrame,
                                                id_column: str,
                                                text_column: str,
                                                class_column: str,
                                                pe_column: str,
                                                replacement_strategy: str,
                                                zero_entity_retain_text: bool,
                                                data_dir_path: Path,
                                                model_file_path: Path) -> Result:
        """
        Redacts private entity tokens in the input_df using the specified 
        replacement strategy, evaluates a text classifier model on the 
        redacted data and returns the evaluation result.
        
        :param input_df: The DataFrame containing the id, text, and class columns.
        :param pe_df: The DataFrame containing the id and private entities columns.
        :param id_column: The column name in the DataFrame that contains the unique identifier for each row.
        :param text_column: The column name in the DataFrame that contains the text.
        :param class_column: The column name in the DataFrame that contains the class labels.
        :param pe_column: The column name in the DataFrame that contains the private entities as list of dictionaries.
        :param replacement_strategy: The strategy for replacing tokens. Default is "semantic_label_mask".
               Options are - 
               (1) "semantic_label_mask"
               (2) "random_mask"
               (3) "generic_mask"
        :param zero_entity_retain_text: If True, retains the original text for rows with zero private entities.
        :param data_dir_path: The directory path where the redacted test CSV will be saved.
        :param model_file_path: The file path to the pre-trained text classifier model.
        
        :return: None
        """

        valid_replacement_strategies = ["semantic_label_mask", "random_mask", "generic_mask"]
        if replacement_strategy in valid_replacement_strategies:
            redacted_df = TokenTreatmentUtils.redact_private_entity_tokens_in_text_for_dataframe_with_pe_df(
                input_df=input_df,
                pe_df=pe_df,
                id_column=id_column,
                text_column=text_column,
                class_column=class_column,
                pe_column=pe_column,
                replacement_strategy=replacement_strategy,
                zero_entity_retain_text=zero_entity_retain_text
            )
            test_filename = f'test_redacted_with_{replacement_strategy}.csv'
        else:
            redacted_df = input_df.copy()
            test_filename = 'test_no_redaction.csv'

        if (data_dir_path / test_filename).exists():
            (data_dir_path / test_filename).unlink()

        redacted_df[
            [
                f'text_redacted_with_{replacement_strategy}' if replacement_strategy in valid_replacement_strategies else text_column,
                class_column
            ]
        ].to_csv(data_dir_path / test_filename,
                 sep='\t',
                 index=False,
                 header=['text', 'label']
        )

        classifier = TextClassifier.load(model_file_path)

        csv.field_size_limit(int(1e8))
        corpus: Corpus = CSVClassificationCorpus(data_folder=data_dir_path,
                                                 column_name_map={
                                                     0: "text",
                                                     1: "label"
                                                 },
                                                 label_type="label",
                                                 train_file='train.csv',
                                                 dev_file='dev.csv',
                                                 test_file=test_filename,
                                                 skip_header=True,
                                                 delimiter="\t")
        
        print(f'Check first row of test corpus:\n{corpus.test[0]}')

        result: Result = classifier.evaluate(corpus.test,
                                             mini_batch_size=1,
                                             gold_label_type="label")
        
        (data_dir_path / test_filename).unlink()

        return result
    
    @staticmethod
    def convert_to_fasttext_format(row: pd.Series,
                                   text_column: str,
                                   class_column: str) -> str:
        """
        Converts a DataFrame row to FastText format.

        :param row: A DataFrame row with text and class columns.
        :param text_column: The name of the text column.
        :param class_column: The name of the class column.
        :return: A string in FastText format.
        """

        labels = " ".join([f"__label__{label.replace(' ', '_')}" for label in row[class_column]])
        text = row[text_column].replace("\n", " ").replace("\t", " ")
        return f"{labels} {text}"
    
    @staticmethod
    def chunk_text_to_fasttext(text: str, 
                               labels: List[str],
                               tokenizer: AutoTokenizer,
                               max_tokens: int = 510) -> List[str]:
        """
        Splits a long text into chunks of MAX_TOKENS, prepends the labels
        in FastText format, returns list of lines for ClassificationCorpus.

        :param text: The input text to be chunked.
        :param labels: List of labels associated with the text.
        :param tokenizer: The tokenizer to use for tokenizing the text.
        :param max_tokens: Maximum number of tokens per chunk, default is 510.
        :return: List of strings, each in FastText format.
        """

        tokens = tokenizer.tokenize(text)
        label_str = " ".join([f"__label__{label.replace(' ', '_')}" for label in labels])
        chunks = list()
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i: i + max_tokens]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(f"{label_str} {chunk_text}")
        return chunks

    @staticmethod
    def write_chunked_fasttext_file(input_df: pd.DataFrame,
                                    file_export_path: Path,
                                    text_column: str,
                                    class_column: str,
                                    tokenizer: AutoTokenizer,
                                    max_tokens: int = 510) -> None:
        """
        Writes a DataFrame to a FastText format file with chunked text.

        :param input_df: DataFrame with text and class columns.
        :param file_export_path: Path to the output file.
        :param text_column: The name of the text column.
        :param class_column: The name of the class column.
        :param tokenizer: The tokenizer to use for tokenizing the text.
        :param max_tokens: Maximum number of tokens per chunk, default is 510.
        :return: None
        """
        
        with open(file_export_path, "w", encoding="utf-8") as f:
            for _, row in tqdm(input_df.iterrows(), 
                               total=len(input_df), 
                               desc=f"Writing chunked FastText format file..."):
                
                text_chunks = EvaluationUtils.chunk_text_to_fasttext(row[text_column], 
                                                                     row[class_column], 
                                                                     tokenizer, 
                                                                     max_tokens)
                for chunk in text_chunks:
                    f.write(chunk + "\n")
    
    @staticmethod
    def redact_and_evaluate_for_mic_mltc(input_df: pd.DataFrame,
                                         pe_df: pd.DataFrame,
                                         id_column: str,
                                         text_column: str,
                                         class_column: str,
                                         pe_column: str,
                                         replacement_strategy: str,
                                         zero_entity_retain_text: bool,
                                         data_dir_path: Path,
                                         model_file_path: Path) -> Result:
        """
        Redacts private entity tokens in the input_df using the specified 
        replacement strategy, evaluates a text classifier model on the 
        redacted data and returns the evaluation result.
        
        :param input_df: The DataFrame containing the id, text, and class columns.
        :param pe_df: The DataFrame containing the id and private entities columns.
        :param id_column: The column name in the DataFrame that contains the unique identifier for each row.
        :param text_column: The column name in the DataFrame that contains the text.
        :param class_column: The column name in the DataFrame that contains the class labels.
        :param pe_column: The column name in the DataFrame that contains the private entities as list of dictionaries.
        :param replacement_strategy: The strategy for replacing tokens. Default is "semantic_label_mask".
               Options are - 
               (1) "semantic_label_mask"
               (2) "random_mask"
               (3) "generic_mask"
        :param zero_entity_retain_text: If True, retains the original text for rows with zero private entities.
        :param data_dir_path: The directory path where the redacted test .txt will be saved.
        :param model_file_path: The file path to the pre-trained text classifier model.
        
        :return: None
        """

        valid_replacement_strategies = ["semantic_label_mask", "random_mask", "generic_mask"]
        if replacement_strategy in valid_replacement_strategies:
            redacted_df = TokenTreatmentUtils.redact_private_entity_tokens_in_text_for_dataframe_with_pe_df(
                input_df=input_df,
                pe_df=pe_df,
                id_column=id_column,
                text_column=text_column,
                class_column=class_column,
                pe_column=pe_column,
                replacement_strategy=replacement_strategy,
                zero_entity_retain_text=zero_entity_retain_text
            )
            test_filename = f'test_redacted_with_{replacement_strategy}.txt'
        else:
            redacted_df = input_df.copy()
            test_filename = 'test_no_redaction.txt'

        if (data_dir_path / test_filename).exists():
            (data_dir_path / test_filename).unlink()
        
        text_column_to_use = (
            f'text_redacted_with_{replacement_strategy}' 
            if replacement_strategy in valid_replacement_strategies
            else text_column
        )

        ### Additional check for BiomedBERT model ###
        transformer_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        if transformer_model_name.replace("/", "--").replace("_", "-") in str(model_file_path):
            tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
            EvaluationUtils.write_chunked_fasttext_file(
                input_df=redacted_df,
                file_export_path=data_dir_path / test_filename,
                text_column=text_column_to_use,
                class_column=class_column,
                tokenizer=tokenizer,
                max_tokens=510
            )
        else:
            with open(data_dir_path / test_filename, "w", encoding="utf-8") as f_train:
                for _, row in tqdm(redacted_df.iterrows(),
                                total=len(redacted_df),
                                desc="Writing FastText format file..."):
                    f_train.write(
                        EvaluationUtils.convert_to_fasttext_format(
                            row, text_column_to_use, class_column
                        ) + "\n"
                    )

        classifier = TextClassifier.load(model_file_path)

        corpus: Corpus = ClassificationCorpus(data_dir_path,
                                              train_file='train.txt',
                                              dev_file='dev.txt',
                                              test_file=test_filename,
                                              label_type=class_column)

        print(f'Check first row of test corpus:\n{corpus.test[0]}')

        evaluate_out_file = f'{test_filename.replace(".txt", ".tsv")}'
        result: Result = classifier.evaluate(corpus.test,
                                             mini_batch_size=1,
                                             gold_label_type=class_column,
                                             out_path=data_dir_path / evaluate_out_file)
        
        (data_dir_path / test_filename).unlink()
        (data_dir_path / evaluate_out_file).unlink()

        return result