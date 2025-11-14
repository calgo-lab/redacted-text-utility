from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from typing import Any, Dict, List

from IPython.display import display

from core.logging import configure_logging, get_logger
from data_handlers.mic_data_handler import MicDataHandler
from infrastructure.services.entity_prediction_service import EntityPredictionService
from utils.project_utils import ProjectUtils
from utils.token_treatment_utils import TokenTreatmentUtils

import json

import pandas as pd


if __name__ == "__main__":
    
    configure_logging()
    
    logger = get_logger(__name__)
    project_root: Path = ProjectUtils.get_project_root()
    data_handler: MicDataHandler = MicDataHandler(project_root)
    
    raw_file_names: List[str] = data_handler.get_available_raw_files()
    logger.info(f"Available raw data files: \n{json.dumps(raw_file_names, indent=2)}")
    
    ### Data files available:
    """
    [
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet"
    ]
    """

    df_dict: Dict[str, pd.DataFrame] = dict()
    for file_name in raw_file_names:
        df = data_handler.get_dataframe_for_file(file_name)
        df_dict[file_name.split("-")[0]] = df
        logger.info(f"Loaded dataframe for file '{file_name}' with {df.shape[0]} rows.")

    """ 
    ### If we decide to use translation service, we need an estimate of total characters to estimate costs
    
    # Count total number of characters in all the texts from all rows of 'text' column in the train dataframe
    total_characters: int = df_dict["train"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the training dataframe: {total_characters}")

    # Count total number of characters in all the texts from all rows of 'text' column in the test dataframe
    total_characters_test: int = df_dict["test"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the test dataframe: {total_characters_test}")

    # Count total number of characters in all the texts from all rows of 'text' column in the validation dataframe
    total_characters_val: int = df_dict["validation"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the validation dataframe: {total_characters_val}")
    """

    ### Log specific row text for checking
    """
    split_name: str = "validation"
    row_id: int = 269
    logger.info(f"df_dict['{split_name}'].row[{row_id}].text:\n{df_dict[split_name].iloc[row_id].text}")
    """

    processed_data_dir: Path = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    """
    entity_prediction_service: EntityPredictionService = EntityPredictionService()
    entity_set_id: str = "ontonotes5"
    model_id: str = "ner-english-ontonotes-large"
    """

    ### Collect named entities for the all splits
    """
    for data_split in ["train", "validation", "test"]:
        output_path = processed_data_dir / f"{data_split}-00000-of-00001_ne.parquet"
        entity_prediction_service.collect_named_entities_for_dataframe(
            entity_set_id=entity_set_id,
            model_id=model_id,
            source_df=df_dict[data_split],
            source_column="text",
            target_column=None,
            target_df_export_path=output_path
        )
        ne_df = pd.read_parquet(output_path)
        named_entities: List[Dict[str, Any]] = list()
        for idx, row in ne_df.iterrows():
            nes = json.loads(row["text_ne_ontonotes5_ner-english-ontonotes-large"])
            [ne.update({"row_idx": idx}) for ne in nes]
            named_entities.extend(nes)
        with open(processed_data_dir / f"named_entities_{data_split}.json", "w", encoding="utf-8") as f:
            json.dump(named_entities, f, indent=2, ensure_ascii=False)
    """

    ### Generate token treatment files for all splits
    """
    for data_split in ["train", "validation", "test"]:
        ne_df = pd.read_parquet(processed_data_dir / f"{data_split}-00000-of-00001_ne.parquet")
        ne_column = "text_ne_ontonotes5_ner-english-ontonotes-large"
        pes, epes = TokenTreatmentUtils.filter_named_entities_for_dataframe(ne_df, ne_column)
        excluded_date_tokens = [ne['token'] for ne in epes if ne['label'] == 'DATE']

        with open(processed_data_dir / f"excluded_date_tokens_{data_split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(excluded_date_tokens))

        with open(processed_data_dir / f"private_entity_tokens_{data_split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(entity['token'] for entity in pes))

        with open(processed_data_dir / f"private_entities_{data_split}.json", "w", encoding="utf-8") as f:
            json.dump(pes, f, indent=2, ensure_ascii=False)
    """

    ### Redact private entities in a specific row of a specific split for checking
    """
    split_name: str = "train"
    ne_df = pd.read_parquet(processed_data_dir / f"{split_name}-00000-of-00001_ne.parquet")
    ne_column = "text_ne_ontonotes5_ner-english-ontonotes-large"
    row_idx: int = 3872
    input_text = ne_df.iloc[row_idx].text
    private_entities, _ = TokenTreatmentUtils.filter_named_entities(
        json.loads(ne_df.iloc[row_idx][ne_column])
    )
    redacted_text = TokenTreatmentUtils.redact_private_entity_tokens_in_text(
        input_text=input_text,
        private_entities=private_entities,
        replacement_strategy="semantic_label_mask"
    )
    logger.info(f"ne_df_{split_name}.row[{row_idx}].text:\n{input_text}\n")
    logger.info(f"private_entities_{split_name}.row[{row_idx}]:\n{private_entities}\n")
    logger.info(f"redacted_text_{split_name}.row[{row_idx}]:\n{redacted_text}\n")
    """

    ### Redact all splits and save to new dataframes
    """ 
    for data_split in ["train", "validation", "test"]:
        output_path = processed_data_dir / f"{data_split}-00000-of-00001_ne_redacted.parquet"
        ne_df = pd.read_parquet(processed_data_dir / f"{data_split}-00000-of-00001_ne.parquet")
        ne_column = "text_ne_ontonotes5_ner-english-ontonotes-large"
        exported_df_path = TokenTreatmentUtils.redact_private_entity_tokens_in_text_for_dataframe(
            ne_df=ne_df,
            text_column="text",
            ne_column=ne_column,
            target_df_export_path=output_path,
            replacement_strategies=[
                "semantic_label_mask",
                "random_mask",
                "generic_mask"
            ]
        )
        logger.info(f"Redacted dataframe for split '{data_split}' exported to: {exported_df_path}")
    """

    ### Log specific row redacted text for checking
    """
    split_name: str = "train"
    ne_df = pd.read_parquet(processed_data_dir / f"{split_name}-00000-of-00001_ne_redacted.parquet")
    row_idx: int = 2106
    original_text = ne_df.iloc[row_idx]["text"]
    redacted_text = ne_df.iloc[row_idx]["text_redacted_with_generic_mask"]
    logger.info(f"pe_redacted_df_{split_name}.row[{row_idx}].original_text:\n{original_text}\n")
    logger.info(f"pe_redacted_df_{split_name}.row[{row_idx}].redacted_text:\n{redacted_text}\n")
    """