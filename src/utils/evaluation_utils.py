from pathlib import Path

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.models import TextClassifier

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
                                                model_file_path: Path,
                                                data_dir_path: Path,
                                                metrics_output_path: Path) -> None:
        """
        Redacts private entity tokens in the input_df using the specified 
        replacement strategy, evaluates a text classifier model on the 
        redacted data and saves the evaluation metrics to the specified 
        output path.
        
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
        :param model_file_path: The file path to the pre-trained text classifier model.
        :param data_dir_path: The directory path where the redacted test CSV will be saved.
        :param metrics_output_path: The file path where the evaluation metrics will be saved.
        
        :return: None
        """

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
        if (data_dir_path / test_filename).exists():
            (data_dir_path / test_filename).unlink()

        redacted_df[
            [
                f'text_redacted_with_{replacement_strategy}',
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

        result = classifier.evaluate(corpus.test,
                                     mini_batch_size=1,
                                     gold_label_type="label")
        
        with open(metrics_output_path, 'w') as f:
            f.write(result.detailed_results)
        
        (data_dir_path / test_filename).unlink()
