from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from typing import Any, Dict, List, Set

from IPython.display import display

from core.logging import configure_logging, get_logger
from data_handlers.echr_data_handler import EchrDataHandler
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
    
    """
    mic_data_handler: MicDataHandler = MicDataHandler(project_root)
    mic_raw_file_names: List[str] = mic_data_handler.get_available_raw_files()
    logger.info(f"Available mic raw data files: \n{json.dumps(mic_raw_file_names, indent=2)}")
    """
    
    ### Data files available:
    """
    [
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet"
    ]
    """

    """
    df_dict: Dict[str, pd.DataFrame] = dict()
    for file_name in mic_raw_file_names:
        df = mic_data_handler.get_dataframe_for_file(file_name)
        df_dict[file_name.split("-")[0]] = df
        logger.info(f"Loaded mic dataframe for file '{file_name}' with {df.shape[0]} rows.")
    """

    ### If we decide to use translation service, we need an estimate of total characters to estimate costs
    """
    # Count total number of characters in all the texts from all rows of 'text' column in the mic train dataframe
    total_characters: int = df_dict["train"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the mic training dataframe: {total_characters}")
    # 721,816

    # Count total number of characters in all the texts from all rows of 'text' column in the mic test dataframe
    total_characters_test: int = df_dict["test"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the mic test dataframe: {total_characters_test}")
    # 103,734

    # Count total number of characters in all the texts from all rows of 'text' column in the mic validation dataframe
    total_characters_val: int = df_dict["validation"]["text"].str.len().sum()
    logger.info(f"Total number of characters in the 'text' column of the mic validation dataframe: {total_characters_val}")
    # 86,937
    """

    ### Log specific row text for checking
    """
    split_name: str = "validation"
    row_id: int = 269
    logger.info(f"df_dict['{split_name}'].row[{row_id}].text:\n{df_dict[split_name].iloc[row_id].text}")
    """

    """
    raw_data_dir: Path = project_root / "data" / "raw" / "DATEXIS" / "med_intent_classification" / "data"
    processed_data_dir: Path = project_root / "data" / "processed" / "DATEXIS" / "med_intent_classification"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    """

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

    ### Calulate statistics on private entities per split
    """
    pe_stats: Dict[str, Dict[str, Any]] = dict()
    for data_split in ["train", "validation", "test"]:
        
        raw_data_file: str = f"{data_split}-00000-of-00001.parquet"
        raw_df: pd.DataFrame = pd.read_parquet(raw_data_dir / raw_data_file)
        total_rows: int = raw_df.shape[0]
        
        with open(processed_data_dir / f"private_entities_{data_split}.json", "r", encoding="utf-8") as f:
            private_entities: List[Dict[str, Any]] = json.load(f)
        
        total_private_entities: int = len(private_entities)
        
        total_rows_with_private_entities: int = len(set(pe['row_idx'] for pe in private_entities))
        
        private_entities_by_label: Dict[str, int] = dict()
        for pe in private_entities:
            label = pe['label']
            private_entities_by_label[label] = private_entities_by_label.get(label, 0) + 1
        
        valid_labels: Set[str] = {"PERSON", "DATE", "GPE", "ORG"}
        for label in valid_labels:
            if label not in private_entities_by_label:
                private_entities_by_label[label] = 0
        
        private_entities_by_label = dict(
            sorted(
                private_entities_by_label.items(), key=lambda item: item[1], reverse=True
            )
        )
        
        pe_stats[raw_data_file] = {
            "T-Rows": total_rows,
            "T-Rows-PE": total_rows_with_private_entities,
            "T-PE": total_private_entities,
            **{f"{label}": count for label, count in private_entities_by_label.items()}
        }

    pe_stats_df = pd.DataFrame.from_dict(pe_stats, orient="index")
    pe_stats_df.index.name = "Data File"
    logger.info(f"\n{pe_stats_df.to_markdown()}")
    """
    
    echr_data_handler: EchrDataHandler = EchrDataHandler(project_root)
    
    echr_raw_file_names: List[str] = echr_data_handler.get_available_raw_files()
    logger.info(f"Available echr raw data files: \n{json.dumps(echr_raw_file_names, indent=2)}")
    
    ### Data files available:
    """
    [
        "ECHR_Dataset.parquet"
    ]
    """

    echr_df: pd.DataFrame = echr_data_handler.get_dataframe_for_file(echr_raw_file_names[0])
    
    
    # row_id: int = 269
    # logger.info(f"echr_df.row[{row_id}].text:\n{echr_df.iloc[row_id].text}")
    # logger.info(f"echr_df.row[{row_id}].binary_judgement:{echr_df.iloc[row_id].binary_judgement}")
   

    ### Total number of rows in echr dataframe
    # logger.info(f"Total number of rows in echr dataframe: {echr_df.shape[0]}")

    ### Print all available column names
    # logger.info(f"echr_df columns: {echr_df.columns.tolist()}")

    ### Print total count of binary_judgement values
    # binary_judgement_counts = echr_df['binary_judgement'].value_counts()
    # logger.info(f"echr_df binary_judgement value counts:\n{binary_judgement_counts}")

    ### Print total count of partition values
    # partition_counts = echr_df['partition'].value_counts()
    # logger.info(f"echr_df partition value counts:\n{partition_counts}")

    ### Unique values in 'itemid' column
    # unique_itemids = echr_df['itemid'].unique()
    # logger.info(f"Unique itemids in echr_df: {len(unique_itemids.tolist())}")

    ### Check how many rows have itemid starting with '001'
    # itemid_starting_with_001 = echr_df[echr_df['itemid'].str.startswith('001')]
    # logger.info(f"Number of rows with itemid starting with '001': {itemid_starting_with_001.shape[0]}")

    ### Get train-dev-test DatasetDict for echr dataset
    # echr_dataset_dict_1 = echr_data_handler.get_train_dev_test_datasetdict(k=1)
    
    ### Check if the itemids in one split overlap with itemids in other splits
    # train_itemids_1 = set(echr_dataset_dict_1['train']['itemid'])
    # dev_itemids_1 = set(echr_dataset_dict_1['dev']['itemid'])
    # test_itemids_1 = set(echr_dataset_dict_1['test']['itemid'])
    # logger.info(f"Overlap between train and dev itemids: {len(train_itemids_1.intersection(dev_itemids_1))}")
    # logger.info(f"Overlap between train and test itemids: {len(train_itemids_1.intersection(test_itemids_1))}")
    # logger.info(f"Overlap between dev and test itemids: {len(dev_itemids_1.intersection(test_itemids_1))}")

    ### Check Overlap between splits of fold 1 and fold 2
    # echr_dataset_dict_2 = echr_data_handler.get_train_dev_test_datasetdict(k=2)
    # train_itemids_2 = set(echr_dataset_dict_2['train']['itemid'])
    # dev_itemids_2 = set(echr_dataset_dict_2['dev']['itemid'])
    # test_itemids_2 = set(echr_dataset_dict_2['test']['itemid'])
    # logger.info(f"Overlap between fold 1 train and fold 2 train itemids: {len(train_itemids_1.intersection(train_itemids_2))}")
    # logger.info(f"Overlap between fold 1 dev and fold 2 dev itemids: {len(dev_itemids_1.intersection(dev_itemids_2))}")
    # logger.info(f"Overlap between fold 1 test and fold 2 test itemids: {len(test_itemids_1.intersection(test_itemids_2))}")

    entity_prediction_service: EntityPredictionService = EntityPredictionService()
    entity_set_id: str = "ontonotes5"
    model_id: str = "ner-english-ontonotes-large"

    """
    model_service = entity_prediction_service._model_service
    mim = model_service.get_model_inference_maker(entity_set_id, model_id)
    result = mim.infer(echr_df.iloc[row_id].text)
    logger.info(f"Named entity recognition result for echr_df.row[{row_id}]:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    """

    processed_data_dir: Path = project_root / "data" / "processed" / "glnmario" / "ECHR"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_data_dir / f"ECHR_Dataset_ne.parquet"
    entity_prediction_service.collect_named_entities_for_dataframe_parallel(
        entity_set_id=entity_set_id,
        model_id=model_id,
        source_df=echr_df,
        source_column="text",
        target_column=None,
        target_df_export_path=output_path
    )
    ne_df = pd.read_parquet(output_path)
    named_entities: List[Dict[str, Any]] = list()
    for _, row in ne_df.iterrows():
        nes = json.loads(row["text_ne_ontonotes5_ner-english-ontonotes-large"])
        [ne.update({"itemid": row["itemid"]}) for ne in nes]
        named_entities.extend(nes)
    with open(processed_data_dir / f"named_entities.json", "w", encoding="utf-8") as f:
        json.dump(named_entities, f, indent=2, ensure_ascii=False)