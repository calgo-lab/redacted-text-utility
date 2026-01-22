from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from typing import Any, Dict, List, Set

from IPython.display import display

from core.logging import configure_logging, get_logger
from data_handlers.echr_data_handler import EchrDataHandler
from data_handlers.mic_data_handler import MicDataHandler
from infrastructure.services.entity_prediction_service import EntityPredictionService
from utils.plot_utils import PlotUtils
from utils.project_utils import ProjectUtils
from utils.report_utils import ReportUtils
from utils.token_treatment_utils import TokenTreatmentUtils

import json

import pandas as pd


if __name__ == "__main__":
    
    configure_logging()
    
    logger = get_logger(__name__)
    project_root: Path = ProjectUtils.get_project_root()
    
    mic_data_handler: MicDataHandler = MicDataHandler(project_root)
    mic_raw_file_names: List[str] = mic_data_handler.get_available_raw_files()
    logger.info(f"Available mic raw data files: \n{json.dumps(mic_raw_file_names, indent=2)}")

    ### Data files available:
    """
    [
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet"
    ]
    """

    df_dict: Dict[str, pd.DataFrame] = dict()
    for file_name in mic_raw_file_names:
        df = mic_data_handler.get_dataframe_for_file(file_name)
        df_dict[file_name.split("-")[0]] = df
        logger.info(f"Loaded mic dataframe for file '{file_name}' with {df.shape[0]} rows.")
    
    ### Find all unique intent labels in the mic dataset
    """
    intents_list = list()
    for intents in df_dict["train"]['intents']:
        intents_list.extend(intents.tolist())

    unique_intents: Set[str] = set(intents_list)
    logger.info(f"Unique intent labels in mic training dataframe ({len(unique_intents)}): \n{json.dumps(sorted(unique_intents), indent=2)}")
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

    ### Filter test set rows with redacted text and analyze intent distribution
    """
    processed_data_dir: Path = project_root / "data" / "processed" / "DATEXIS" / "med_intent_classification"
    split_name: str = "test"
    test_ne_df = pd.read_parquet(processed_data_dir / f"{split_name}-00000-of-00001_ne_redacted.parquet")
    
    filtered_test_ne_df = test_ne_df[
        (test_ne_df["text_redacted_with_semantic_label_mask"].notnull()) &
        (test_ne_df["text_redacted_with_semantic_label_mask"] != "")
    ]
    logger.info(f"Number of rows in filtered_test_ne_df : {filtered_test_ne_df.shape[0]}")

    filtered_test_intents = list()
    for intents in filtered_test_ne_df['intents']:
        filtered_test_intents.extend(intents.tolist())
    
    unique_filtered_test_intents: Set[str] = set(filtered_test_intents)
    filtered_test_intents_counts: Dict[str, int] = dict()
    for intent in unique_filtered_test_intents:
        filtered_test_intents_counts[intent] = filtered_test_intents.count(intent)
    
    logger.info(f"Frequency of filtered test intents ({len(filtered_test_intents_counts)}): \n{json.dumps(filtered_test_intents_counts, indent=2)}")
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
    
    ### Log specific row text for checking
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

    ### Get num_tokens dataframe for echr dataset and log statistics
    """
    num_tokens_df = echr_data_handler.get_num_tokens_df(echr_raw_file_names[0])
    logger.info(f"num_tokens statistics:\n{num_tokens_df['num_tokens'].describe()}")
    
    ## OUTPUT:
    ## count    11478
    ## mean      2538.018470
    ## std       2924.495029
    ## min         14
    ## 25%        818
    ## 50%       1737
    ## 75%       3184
    ## max      59784
    
    ### Plot num_tokens distribution
    plot_output_path = project_root / "plots" / "glnmario" / "ECHR" / "eda"
    plot_output_path.mkdir(parents=True, exist_ok=True)
    plot_file_name = f"{echr_raw_file_names[0].replace('.parquet', '')}_num_tokens_distribution.jpg"
    PlotUtils.plot_num_tokens_distribution(num_tokens_df, 
                                           plot_output_path / plot_file_name,
                                           plot_config={"bins": 50, "show_grid": True})
    
    num_tokens_df_filtered = num_tokens_df[
        (num_tokens_df["num_tokens"] >= 512) & (num_tokens_df["num_tokens"] <= 5120)
    ]
    logger.info(f"Number of documents with total tokens between 512 and 5120: {num_tokens_df_filtered.shape[0]}")

    filtered_echr_df = echr_df[echr_df["itemid"].isin(num_tokens_df_filtered["itemid"])]
    logger.info(f"binary_judgement distribution:\n{filtered_echr_df['binary_judgement'].value_counts()}")
    """

    ### Get NER result for specific row for checking
    """
    entity_prediction_service: EntityPredictionService = EntityPredictionService()
    entity_set_id: str = "ontonotes5"
    model_id: str = "ner-english-ontonotes-large"
    row_id: int = 0
    model_service = entity_prediction_service._model_service
    mim = model_service.get_model_inference_maker(entity_set_id, model_id)
    result = mim.infer(echr_df.iloc[row_id].text)
    logger.info(f"Named entity recognition result for echr_df.row[{row_id}]:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    """

    ### Collect named entities for echr dataframe
    """
    entity_prediction_service: EntityPredictionService = EntityPredictionService()
    entity_set_id: str = "ontonotes5"
    model_id: str = "ner-english-ontonotes-large"
    
    processed_data_dir: Path = project_root / "data" / "processed" / "glnmario" / "ECHR"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / f"ECHR_Dataset_ne.parquet"
    
    entity_prediction_service.collect_named_entities_for_dataframe(
        entity_set_id=entity_set_id,
        model_id=model_id,
        source_df=echr_df,
        source_column="text",
        target_column=None,
        target_df_export_path=output_path,
        export_with_only_id_column="itemid"
    )
    
    ne_df = pd.read_parquet(output_path)
    named_entities: List[Dict[str, Any]] = list()
    for _, row in ne_df.iterrows():
        nes = json.loads(row["text_ne_ontonotes5_ner-english-ontonotes-large"])
        [ne.update({"itemid": row["itemid"]}) for ne in nes]
        named_entities.extend(nes)
    with open(processed_data_dir / f"named_entities_partial.json", "w", encoding="utf-8") as f:
        json.dump(named_entities, f, indent=2, ensure_ascii=False)
    """

    ### Collect private entities for echr dataframe
    """ 
    processed_data_dir: Path = project_root / "data" / "processed" / "glnmario" / "ECHR"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    ne_df = pd.read_parquet(processed_data_dir / f"ECHR_Dataset_ne.parquet")
    ne_column = "text_ne_ontonotes5_ner-english-ontonotes-large"
    target_column = "text_pe_ontonotes5_ner-english-ontonotes-large"
    target_df_export_path = processed_data_dir / f"ECHR_Dataset_pe.parquet"
    TokenTreatmentUtils.collect_private_entity_entities_for_dataframe(
        ne_df=ne_df,
        ne_column=ne_column,
        target_column=target_column,
        target_df_export_path=target_df_export_path
    )
    """

    ### Analyze private entities dataframe
    """
    pe_df = echr_data_handler.get_private_entities_df(echr_raw_file_names[0])

    num_rows_with_zero_pe = pe_df[pe_df["text_pe_ontonotes5_ner-english-ontonotes-large"] == "[]"].shape
    logger.info(f"Number of rows with zero private entities: {num_rows_with_zero_pe[0]}")

    num_tokens_df = echr_data_handler.get_num_tokens_df(echr_raw_file_names[0])
    num_tokens_df_filtered = num_tokens_df[
        (num_tokens_df["num_tokens"] >= 512) & (num_tokens_df["num_tokens"] <= 5120)
    ]
    pe_df_filtered = pe_df[pe_df["itemid"].isin(num_tokens_df_filtered["itemid"])]

    num_rows_with_zero_pe = pe_df_filtered[pe_df_filtered["text_pe_ontonotes5_ner-english-ontonotes-large"] == "[]"].shape
    logger.info(f"Number of rows with zero private entities for pe_df_filtered: {num_rows_with_zero_pe[0]}")
    """

    ### Update private entities stat into private entities dataframe
    """
    processed_data_dir: Path = project_root / "data" / "processed" / "glnmario" / "ECHR"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    pe_df_file_path = processed_data_dir / f"ECHR_Dataset_pe.parquet"
    pe_column = "text_pe_ontonotes5_ner-english-ontonotes-large"
    id_column = "itemid"

    stats = TokenTreatmentUtils.update_private_entity_dataframe_with_stats(
        pe_df_file_path=pe_df_file_path,
        pe_column=pe_column,
        id_column=id_column
    )
    logger.info(f"Private entity statistics:\n{json.dumps(stats, indent=2)}")
    """

    ### Get updated private entities dataframe for echr dataset
    """
    pe_df = echr_data_handler.get_private_entities_df(echr_raw_file_names[0])
    logger.info(f"pe_df columns: {pe_df.columns.tolist()}")
    """

    ### Get private entity statistics for filtered echr private entities dataframe
    """
    pe_df = echr_data_handler.get_private_entities_df(echr_raw_file_names[0])
    num_tokens_df = echr_data_handler.get_num_tokens_df(echr_raw_file_names[0])
    num_tokens_df_filtered = num_tokens_df[
        (num_tokens_df["num_tokens"] >= 512) & (num_tokens_df["num_tokens"] <= 5120)
    ]
    pe_df_filtered = pe_df[pe_df["itemid"].isin(num_tokens_df_filtered["itemid"])]
    stats = echr_data_handler.get_private_entity_stats(pe_df_filtered)
    logger.info(f"Private entity statistics for filtered pe_df:\n{json.dumps(stats, indent=2)}")
    """

    ### Get train-dev-test DatasetDict for echr dataset for k=1
    """ 
    echr_dataset_dict = echr_data_handler.get_train_dev_test_datasetdict(k=1)
   
    ### Log number of rows in each split
    for split_name, dataset in echr_dataset_dict.items():
        logger.info(f"Number of rows in echr_dataset_dict['{split_name}']: {len(dataset)}")
    
    ### Total number of rows in all splits
    total_rows_all_splits = sum(len(dataset) for dataset in echr_dataset_dict.values())
    logger.info(f"Total number of rows in all splits: {total_rows_all_splits}")

    ### Get fold stats for echr dataset for k=1
    fold_stats = echr_data_handler.get_fold_stats(echr_dataset_dict)
    logger.info(f"ECHR Dataset k=1 fold stats:\n{json.dumps(fold_stats, indent=2)}")
    """

    ### Redact private entities in the echr test set for k=1 and log redacted dataframe
    """
    redacted_df = TokenTreatmentUtils.redact_private_entity_tokens_in_text_for_dataframe_with_pe_df(
        input_df=echr_data_handler.get_train_dev_test_datasetdict(k=1)["test"].to_pandas(),
        pe_df=echr_data_handler.get_private_entities_df(echr_raw_file_names[0]),
        id_column="itemid",
        text_column="text",
        class_column="binary_judgement",
        pe_column="text_pe_ontonotes5_ner-english-ontonotes-large",
        replacement_strategy="semantic_label_mask",
        zero_entity_retain_text=True
    )
    logger.info(f"Redacted dataframe columns: {redacted_df.columns.tolist()}")
    logger.info(redacted_df.iloc[0].to_dict())
    """

    ### Get performance metrics with hierarchy for echr tc experiments
    """
    reports = ReportUtils.get_performance_metrics_with_hierarchy(
        metrics_dir=project_root / "metrics" / "glnmario" / "ECHR" / "tc"
    )
    logger.info(f'{list(reports.keys())[0]}: {json.dumps(reports.get(list(reports.keys())[0]), indent=2)}')
    """
    
    ### Generate performance metrics summary table (redaction strategies vs models)
    """
    table_report = ReportUtils.get_echr_tc_performance_metrics_summary_table(
        metrics_dir=project_root / "metrics" / "glnmario" / "ECHR" / "tc",
        row_dimension='redaction_strategy',
        row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
        column_dimension='model_name',
        column_values_order=['xlm-roberta-large', 'bert-large-cased', 'google--electra-large-discriminator'],
        class_or_stat='macro avg',
        hierarchy=['model_name', 'redaction_strategy', 'percentile', 'metric_type'],
        fixed_dimensions={'percentile': '0-100', 'metric_type': 'f1-score'},
        model_name_aliases={
            'xlm-roberta-large': 'xlm-roberta-large',
            'bert-large-cased': 'bert-large-cased',
            'google--electra-large-discriminator': 'electra-large-discriminator'
        },
        redaction_strategy_aliases={
            'unredacted': 'No Redaction',
            'semantic_label_mask': 'Semantic Label Masking',
            'random_mask': 'Random Masking',
            'generic_mask': 'Generic Masking'
        }
    )
    logger.info(f"Performance metrics summary table:\n{table_report.to_markdown()}")
    """

    ### Get private entity counts statistics across all folds
    """
    k_range = range(1, 6)
    pe_df = echr_data_handler.get_private_entities_df(echr_raw_file_names[0])
    fold_stats_dict: Dict[int, pd.Series] = dict()
    for k in k_range:
        test_df_k = echr_data_handler.get_train_dev_test_datasetdict(k=k)["test"].to_pandas()
        pe_df_filtered = pe_df[pe_df["itemid"].isin(test_df_k["itemid"])]
        fold_stats_dict[k] = pe_df_filtered['pe_count_total'].describe()
    
    entity_counts_stats_table = pd.DataFrame(fold_stats_dict).T
    entity_counts_stats_table = entity_counts_stats_table.astype(int)
    entity_counts_stats_table.index.name = "Fold (k)"
    logger.info(f"Private entity counts statistics across all folds:\n{entity_counts_stats_table.to_markdown()}")
    """
    
    ### Prepare entity-count / token-count stats for all percentile ranges
    """
    pe_df = echr_data_handler.get_private_entities_df(echr_raw_file_names[0])
    num_tokens_df = echr_data_handler.get_num_tokens_df(echr_raw_file_names[0])
    percentile_ranges = {
        "0-100": (0, 100), 
        "0-25": (0, 25), 
        "25-50": (25, 50), 
        "50-75": (50, 75), 
        "75-100": (75, 100)
    }
    k_range = range(1, 6)

    percentile_token_count_stats_dict = dict()
    percentile_entity_count_stats_dict = dict()
    for k in k_range:
        test_df_k = echr_data_handler.get_train_dev_test_datasetdict(k=k)["test"].to_pandas()
        pe_df_k = pe_df[pe_df["itemid"].isin(test_df_k["itemid"])]
        
        for percentile_label, (p_min, p_max) in percentile_ranges.items():
            lb = int(pe_df_k['pe_count_total'].quantile(q=p_min/100))
            ub = int(pe_df_k['pe_count_total'].quantile(q=p_max/100))
            
            pe_df_k_filtered = pe_df_k[
                (pe_df_k['pe_count_total'] >= lb) &
                (pe_df_k['pe_count_total'] <= ub)
            ]
            entity_count_stats_k = pe_df_k_filtered['pe_count_total'].describe()
            
            stat_keys = entity_count_stats_k.to_dict().keys()

            entity_count_stats_dict = percentile_entity_count_stats_dict.get(
                percentile_label, {stat_key: list() for stat_key in stat_keys}
            )
            for stat_key in stat_keys:
                entity_count_stats_dict[stat_key].append(entity_count_stats_k[stat_key])
            percentile_entity_count_stats_dict[percentile_label] = entity_count_stats_dict


            num_tokens_df_filtered = num_tokens_df[
                num_tokens_df["itemid"].isin(pe_df_k_filtered["itemid"])
            ]            
            token_count_stats_k = num_tokens_df_filtered['num_tokens'].describe()
                        
            token_count_stats_dict = percentile_token_count_stats_dict.get(
                percentile_label, {stat_key: list() for stat_key in stat_keys}
            )
            for stat_key in stat_keys:
                token_count_stats_dict[stat_key].append(token_count_stats_k[stat_key])
            percentile_token_count_stats_dict[percentile_label] = token_count_stats_dict
    
    def average_stats_across_folds(stats_dict):
        return {
            stat_key: sum(values) / len(values)
            for stat_key, values in stats_dict.items()
        }

    combined_stats_table = dict()
    for percentile_label in percentile_token_count_stats_dict.keys():
        avg_token_stats = average_stats_across_folds(
            percentile_token_count_stats_dict[percentile_label]
        )
        avg_entity_stats = average_stats_across_folds(
            percentile_entity_count_stats_dict[percentile_label]
        )

        row = dict()
        for stat_key in avg_token_stats.keys():
            if stat_key == "count":
                row["total_item"] = int(round(avg_token_stats["count"]))
            else:
                row[stat_key] = (
                    f"{int(round(avg_entity_stats[stat_key]))} / "
                    f"{int(round(avg_token_stats[stat_key]))}"
                )

        combined_stats_table[percentile_label] = row
    
    combined_df = (
        pd.DataFrame.from_dict(combined_stats_table, orient="index").loc[
            :, ["total_item", "mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
    )

    logger.info(f"Entity count / Token count statistics across all folds for percentile ranges:\n{combined_df.to_markdown()}")
    """

    ### Generate performance metrics summary table (redaction strategies vs percentiles)
    """
    table_report = ReportUtils.get_echr_tc_performance_metrics_summary_table(
        metrics_dir=project_root / "metrics" / "glnmario" / "ECHR" / "tc",
        row_dimension='redaction_strategy',
        row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
        column_dimension='percentile',
        column_values_order=["0-100", "0-25", "25-50", "50-75", "75-100"],
        class_or_stat='macro avg',
        hierarchy=['model_name', 'redaction_strategy', 'percentile', 'metric_type'],
        fixed_dimensions={'model_name': 'xlm-roberta-large', 'metric_type': 'f1-score'},
        model_name_aliases={
            'xlm-roberta-large': 'xlm-roberta-large',
            'bert-large-cased': 'bert-large-cased',
            'google--electra-large-discriminator': 'electra-large-discriminator'
        },
        redaction_strategy_aliases={
            'unredacted': 'No Redaction',
            'semantic_label_mask': 'Semantic Label Masking',
            'random_mask': 'Random Masking',
            'generic_mask': 'Generic Masking'
        }
    )
    logger.info(f"Performance metrics summary table:\n{table_report.to_markdown()}")
    """

    ### Plot macro F1-score by entity percentiles for all models and redaction strategies
    """
    table_reports = list()
    for model_name in ['xlm-roberta-large', 'bert-large-cased', 'google--electra-large-discriminator']:
        table_report = ReportUtils.get_echr_tc_performance_metrics_summary_table(
            metrics_dir=project_root / "metrics" / "glnmario" / "ECHR" / "tc",
            row_dimension='redaction_strategy',
            row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
            column_dimension='percentile',
            column_values_order=["0-100", "0-25", "25-50", "50-75", "75-100"],
            class_or_stat='macro avg',
            hierarchy=['model_name', 'redaction_strategy', 'percentile', 'metric_type'],
            fixed_dimensions={'model_name': model_name, 'metric_type': 'f1-score'},
            model_name_aliases={
                'xlm-roberta-large': 'xlm-roberta-large',
                'bert-large-cased': 'bert-large-cased',
                'google--electra-large-discriminator': 'electra-large-discriminator'
            },
            redaction_strategy_aliases={
                'unredacted': 'No Redaction',
                'semantic_label_mask': 'Semantic Label Masking',
                'random_mask': 'Random Masking',
                'generic_mask': 'Generic Masking'
            }
        )
        table_reports.append(table_report)

    strategy_styles = {
        "No Redaction": {
            "color": "olivedrab",
            "marker": "o"
        },
        "Semantic Label Masking": {
            "color": "firebrick",
            "marker": "o"
        },
        "Random Masking": {
            "color": "cornflowerblue",
            "marker": "o"
        },
        "Generic Masking": {
            "color": "darkgoldenrod",
            "marker": "o"
        }
    }
    
    fig, ax = PlotUtils.plot_echr_tc_classifier_performance_type_one(
        model_dfs=table_reports,
        model_names=[
            "xlm-roberta-large",
            "bert-large-cased",
            "electra-large-discriminator"
        ],
        percentile_labels=["0-100", "0-25", "25-50", "50-75", "75-100"],
        entity_counts={
            "0-100": 1687,
            "0-25": 433,
            "25-50": 435,
            "50-75": 439,
            "75-100": 427
        },
        redaction_strategies=list(strategy_styles.keys()),
        model_bg_colors={
            "xlm-roberta-large": "peachpuff",
            "bert-large-cased": "palegreen",
            "electra-large-discriminator": "lightpink"
        },
        strategy_styles=strategy_styles
    )

    figure_dir = project_root / "plots" / "glnmario" / "ECHR" / "tc"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_dir / "macro_f1_models_vs_entity_percentiles_redaction_strategies.png",
        dpi=300,
        bbox_inches="tight"
    ) 
    """

    ### Plot macro F1-score comparison between No Redaction and other redaction strategies
    
    model_name = 'google--electra-large-discriminator'
    model_name_alias = 'electra-large-discriminator'
    redaction_strategy = 'generic_mask'
    redaction_strategy_alias = 'Generic Masking'
    
    model_performance = ReportUtils.get_echr_tc_performance_metrics_summary_table(
        metrics_dir=project_root / "metrics" / "glnmario" / "ECHR" / "tc",
        row_dimension='redaction_strategy',
        row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
        column_dimension='percentile',
        column_values_order=["0-100", "0-25", "25-50", "50-75", "75-100"],
        class_or_stat='macro avg',
        hierarchy=['model_name', 'redaction_strategy', 'percentile', 'metric_type'],
        fixed_dimensions={'model_name': model_name, 'metric_type': 'f1-score'},
        model_name_aliases={
            'xlm-roberta-large': 'xlm-roberta-large',
            'bert-large-cased': 'bert-large-cased',
            'google--electra-large-discriminator': 'electra-large-discriminator'
        },
        redaction_strategy_aliases={
            'unredacted': 'No Redaction',
            'semantic_label_mask': 'Semantic Label Masking',
            'random_mask': 'Random Masking',
            'generic_mask': 'Generic Masking'
        }
    )

    fig, ax = PlotUtils.plot_echr_tc_classifier_performance_type_two(
        model_df=model_performance,
        model_name=model_name_alias,
        strategy_a="No Redaction",
        strategy_b=redaction_strategy_alias,
        percentile_labels=["0-100", "0-25", "25-50", "50-75", "75-100"],
        colors={
            "No Redaction": "olivedrab",
            "Semantic Label Masking": "firebrick",
            "Random Masking": "cornflowerblue",
            "Generic Masking": "darkgoldenrod",
        }
    )
    figure_dir = project_root / "plots" / "glnmario" / "ECHR" / "tc" / model_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_dir / f"macro_f1_{model_name}_no_redaction_vs_{redaction_strategy}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    