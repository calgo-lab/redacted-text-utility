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

import numpy as np
import pandas as pd


if __name__ == "__main__":
    
    configure_logging()
    
    logger = get_logger(__name__)
    project_root: Path = ProjectUtils.get_project_root()

    def make_json_serializable(obj) -> Any:
        """
        Converts an object to a JSON-serializable format.
        
        :param obj: The object to convert.
        :return: A JSON-serializable representation of the object.
        """

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, (dict, list, int, float, bool)) or obj is None:
            return obj
        
        if isinstance(obj, str):
            obj_strip = obj.strip()
            if (obj_strip.startswith("[") and obj_strip.endswith("]")) or \
            (obj_strip.startswith("{") and obj_strip.endswith("}")):
                try:
                    return json.loads(obj)
                except json.JSONDecodeError:
                    pass
        
        return obj
    
    mic_data_handler: MicDataHandler = MicDataHandler(project_root)
    
    ### Data files available:
    """
    mic_raw_file_names: List[str] = mic_data_handler.get_available_raw_files()
    logger.info(f"Available mic raw data files: \n{json.dumps(mic_raw_file_names, indent=2)}")

    ### [
    ###     "train-00000-of-00001.parquet",
    ###     "validation-00000-of-00001.parquet",
    ###     "test-00000-of-00001.parquet"
    ### ]
    """

    ### Get merged dataframe for mic dataset and log basic info
    """
    mic_merged_df = mic_data_handler.get_merged_dataframe()
    logger.info(f"mic_merged_df.shape: {mic_merged_df.shape}")
    logger.info(f"mic_merged_df.columns: {mic_merged_df.columns.tolist()}")
    first_row_dict = {
        k: make_json_serializable(v)
        for k, v in mic_merged_df.iloc[0].to_dict().items()
    }
    logger.info(
        f"mic_merged_df[0]:\n{json.dumps(first_row_dict, indent=2, ensure_ascii=False)}"
    )
    """

    ### Get num_tokens dataframe for mic dataset and log statistics
    """
    num_tokens_df_mic = mic_data_handler.get_num_tokens_df()
    logger.info(f"num_tokens statistics for mic_merged_df:\n{num_tokens_df_mic['num_tokens'].describe()}")

    percentile = 0.9983
    logger.info(f"{round(percentile*100, 4)}th percentile of num_tokens in mic_merged_df: {num_tokens_df_mic['num_tokens'].quantile(q=percentile)}")
    logger.info(f"Number of rows with num_tokens > 512: {len(num_tokens_df_mic[num_tokens_df_mic['num_tokens'] > 512])}")
    
    ### Plot num_tokens distribution
    plot_output_path = project_root / "plots" / "DATEXIS" / "med_intent_classification" / "eda"
    plot_output_path.mkdir(parents=True, exist_ok=True)
    plot_file_name = f"mic_merged_num_tokens_distribution.jpg"
    PlotUtils.plot_num_tokens_distribution(num_tokens_df_mic, 
                                           plot_output_path / plot_file_name,
                                           plot_config={"bins": 50, "show_grid": True})
    """
    
    ### Create intent distribution table for mic dataset
    """
    mic_merged_df = mic_data_handler.get_merged_dataframe()
    intent_distribution_dict: Dict[str, int] = dict()
    for _, row in mic_merged_df.iterrows():
        intents = row['intents']
        for intent in intents:
            intent_distribution_dict[intent] = intent_distribution_dict.get(intent, 0) + 1
    
    intent_distribution_dict = sorted(intent_distribution_dict.items(), key=lambda item: item[0])
    intent_distribution_df = pd.DataFrame.from_dict(intent_distribution_dict)
    intent_distribution_df.columns = ['Intent', 'Count']
    logger.info(f"Intent distribution table for mic dataset:\n{intent_distribution_df.to_markdown(index=False)}")
    """

    ### Plot intent distribution bar chart
    """
    plot_output_path = project_root / "plots" / "DATEXIS" / "med_intent_classification" / "eda"
    plot_output_path.mkdir(parents=True, exist_ok=True)
    plot_file_name = "mic_intent_distribution_bar_chart.jpg"
    PlotUtils.plot_mic_intent_distribution_bar_chart(intent_distribution_df, plot_output_path / plot_file_name)
    """

    ### Collect named entities for mic dataset
    """
    mic_merged_df = mic_data_handler.get_merged_dataframe()

    entity_prediction_service: EntityPredictionService = EntityPredictionService()
    entity_set_id: str = "ontonotes5"
    model_id: str = "ner-english-ontonotes-large"
    
    processed_data_dir: Path = project_root / "data" / "processed" / "DATEXIS" / "med_intent_classification"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / f"mic_merged_ne.parquet"
    
    entity_prediction_service.collect_named_entities_for_dataframe(
        entity_set_id=entity_set_id,
        model_id=model_id,
        source_df=mic_merged_df,
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
    with open(processed_data_dir / f"named_entities.json", "w", encoding="utf-8") as f:
        json.dump(named_entities, f, indent=2, ensure_ascii=False)
    """

    ### Collect private entities for mic dataset
    """
    processed_data_dir: Path = project_root / "data" / "processed" / "DATEXIS" / "med_intent_classification"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    ne_df = pd.read_parquet(processed_data_dir / f"mic_merged_ne.parquet")
    ne_column = "text_ne_ontonotes5_ner-english-ontonotes-large"
    target_column = "text_pe_ontonotes5_ner-english-ontonotes-large"
    target_df_export_path = processed_data_dir / f"mic_merged_pe.parquet"
    TokenTreatmentUtils.collect_private_entity_entities_for_dataframe(
        ne_df=ne_df,
        ne_column=ne_column,
        target_column=target_column,
        target_df_export_path=target_df_export_path
    )
    """

    ### Update private entities stat into private entities dataframe
    """
    processed_data_dir: Path = project_root / "data" / "processed" / "DATEXIS" / "med_intent_classification"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    pe_df_file_path = processed_data_dir / f"mic_merged_pe.parquet"
    pe_column = "text_pe_ontonotes5_ner-english-ontonotes-large"
    id_column = "itemid"

    stats = TokenTreatmentUtils.update_private_entity_dataframe_with_stats(
        pe_df_file_path=pe_df_file_path,
        pe_column=pe_column,
        id_column=id_column
    )
    logger.info(f"Private entity statistics:\n{json.dumps(stats, indent=2)}")
    """

    ### Load and analyze private entities dataframe for mic dataset
    """
    mic_merged_pe_df = mic_data_handler.get_private_entities_df()
    logger.info(f"mic_merged_pe_df columns: {mic_merged_pe_df.columns.tolist()}")
    ### [
    ###     'itemid', 
    ###     'text_pe_ontonotes5_ner-english-ontonotes-large', 
    ###     'pe_count_total', 
    ###     'pe_count_ORG', 
    ###     'pe_count_PERSON', 
    ###     'pe_count_DATE', 
    ###     'pe_count_GPE'
    ### ]

    mic_merged_df = mic_data_handler.get_merged_dataframe()
    entity_stats = mic_data_handler.get_private_entity_stats(mic_merged_df)
    logger.info(f"Private entity statistics for mic_merged_df:\n{json.dumps(entity_stats, indent=2)}")

    mic_merged_pe_df_filtered = mic_merged_pe_df[
        mic_merged_pe_df["text_pe_ontonotes5_ner-english-ontonotes-large"] != "[]"
    ]
    logger.info(f"Number of rows in mic_merged_pe_df_filtered with non-empty private entities: {mic_merged_pe_df_filtered.shape[0]}")
    first_row_dict = {
        k: make_json_serializable(v)
        for k, v in mic_merged_pe_df_filtered.iloc[0].to_dict().items()
    }
    logger.info(f"First row of mic_merged_pe_df_filtered:\n{json.dumps(first_row_dict, indent=2, ensure_ascii=False)}")
    """

    ### Get k-fold DatasetDict for mic dataset and check - 
    ### - if in a single fold the train, dev, test sets are mutually exclusive
    ### - if across all folds the test sets are mutually exclusive
    ### - if across all folds the train sets have at most 67% common itemids
    """
    for k in range(1, 6):
        mic_dataset_dict = mic_data_handler.get_train_dev_test_datasetdict(k=k)
        train_ids: Set[str] = set(mic_dataset_dict["train"]["itemid"])
        dev_ids: Set[str] = set(mic_dataset_dict["dev"]["itemid"])
        test_ids: Set[str] = set(mic_dataset_dict["test"]["itemid"])
        assert train_ids.isdisjoint(dev_ids), f"Train and Dev sets are not mutually exclusive for fold {k}"
        assert train_ids.isdisjoint(test_ids), f"Train and Test sets are not mutually exclusive for fold {k}"
        assert dev_ids.isdisjoint(test_ids), f"Dev and Test sets are not mutually exclusive for fold {k}"
    
    for k1, k2 in [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]:
        mic_dataset_dict_k1 = mic_data_handler.get_train_dev_test_datasetdict(k=k1)
        mic_dataset_dict_k2 = mic_data_handler.get_train_dev_test_datasetdict(k=k2)
        test_ids_k1: Set[str] = set(mic_dataset_dict_k1["dev"]["itemid"])
        test_ids_k2: Set[str] = set(mic_dataset_dict_k2["dev"]["itemid"])
        assert test_ids_k1.isdisjoint(test_ids_k2), f"Test sets are not mutually exclusive between fold {k1} and fold {k2}"
    
    for k1, k2 in [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]:
        mic_dataset_dict_k1 = mic_data_handler.get_train_dev_test_datasetdict(k=k1)
        mic_dataset_dict_k2 = mic_data_handler.get_train_dev_test_datasetdict(k=k2)
        train_ids_k1: Set[str] = set(mic_dataset_dict_k1["train"]["itemid"])
        train_ids_k2: Set[str] = set(mic_dataset_dict_k2["train"]["itemid"])
        common_ids = train_ids_k1.intersection(train_ids_k2)
        common_percentage = (len(common_ids) / min(len(train_ids_k1), len(train_ids_k2))) * 100
        assert common_percentage <= 67.0, f"Common percentage of itemids between train sets of fold {k1} and fold {k2} exceeds 67%: {common_percentage}%"
    """

    ### Check number of unique intents in mic dataset train, dev, test sets across all folds
    """
    for k in range(1, 6):
        mic_dataset_dict = mic_data_handler.get_train_dev_test_datasetdict(k=k)
        for split_name in ["train", "dev", "test"]:
            dataset_split = mic_dataset_dict[split_name]
            unique_intents: Set[str] = set()
            for intents_list in dataset_split["intents"]:
                unique_intents.update(intents_list)
            logger.info(f"Fold {k} - {split_name} set has {len(unique_intents)} unique intents.")
    """

    ### Get fold stats for mic dataset for provided k
    """
    k = 1
    kth_datasetdict = mic_data_handler.get_train_dev_test_datasetdict(k=k)
    fold_stats = mic_data_handler.get_fold_stats(kth_datasetdict)
    logger.info(f"mic dataset k={k} fold stats:\n{json.dumps(fold_stats, indent=2)}")
    """

    ### Generate fold-wise performance metrics table with Macro F1-score for 
    ### medical intent multi-label classification task for a specific model
    """
    reports = ReportUtils.get_performance_metrics_with_hierarchy(
        metrics_dir=project_root / "metrics" / "DATEXIS" / "med_intent_classification" / "mltc"
    )
    model_name = "microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"
    label_or_stat = "macro avg"
    metric_index = 2
    redaction_strategy_order = ['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask']
    model_data = reports[model_name]
    table_data = dict()
    for strategy in redaction_strategy_order:
        fold_scores = list()
        for fold_num in range(0, 5):
            f1_score = model_data[strategy][label_or_stat][fold_num][metric_index]
            fold_scores.append(f1_score)
        table_data[strategy] = fold_scores
    
    df_index_name_mapping = {
        'unredacted': 'No Redaction',
        'semantic_label_mask': 'Semantic Label Masking',
        'random_mask': 'Random Masking',
        'generic_mask': 'Generic Masking'
    }
    df = pd.DataFrame(table_data, index=[f'Fold {i}' for i in range(1, 6)]).T
    df.index.name = 'Redaction Strategy'
    df = df.rename(index=df_index_name_mapping)
    df.columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    markdown_table = f"Model: {model_name}\nMetric: Macro F1-score\n\n"
    markdown_table += df.to_markdown()
    logger.info(f"\n{markdown_table}")
    """

    ### Generate performance metrics summary table (redaction strategies vs models)
    """
    table_report = ReportUtils.get_performance_metrics_summary_table(
        metrics_dir=project_root / "metrics" / "DATEXIS" / "med_intent_classification" / "mltc",
        row_dimension='redaction_strategy',
        row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
        column_dimension='model_name',
        column_values_order=['xlm-roberta-large', 'bert-large-cased', 'microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract'],
        class_or_stat='macro avg',
        hierarchy=['model_name', 'redaction_strategy', 'metric_type'],
        fixed_dimensions={'metric_type': 'f1-score'},
        model_name_aliases={
            'xlm-roberta-large': 'xlm-roberta-large',
            'bert-large-cased': 'bert-large-cased',
            'microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract': 'pubmedbert-base-uncased'
        },
        redaction_strategy_aliases={
            'unredacted': 'No Redaction',
            'semantic_label_mask': 'Semantic Label Masking',
            'random_mask': 'Random Masking',
            'generic_mask': 'Generic Masking'
        }
    )
    table_report.index.name = "Redaction Strategy"
    logger.info(f"Performance metrics summary table:\n{table_report.to_markdown()}")
    """

    ### Plot macro F1-score by entity percentiles for all models and redaction strategies
    """
    table_report = ReportUtils.get_performance_metrics_summary_table(
        metrics_dir=project_root / "metrics" / "DATEXIS" / "med_intent_classification" / "mltc",
        row_dimension='redaction_strategy',
        row_values_order=['unredacted', 'semantic_label_mask', 'random_mask', 'generic_mask'],
        column_dimension='model_name',
        column_values_order=['xlm-roberta-large', 'bert-large-cased', 'microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract'],
        class_or_stat='macro avg',
        hierarchy=['model_name', 'redaction_strategy', 'metric_type'],
        fixed_dimensions={'metric_type': 'f1-score'},
        model_name_aliases={
            'xlm-roberta-large': 'xlm-roberta-large',
            'bert-large-cased': 'bert-large-cased',
            'microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract': 'pubmedbert-base-uncased'
        },
        redaction_strategy_aliases={
            'unredacted': 'No Redaction',
            'semantic_label_mask': 'Semantic Label Masking',
            'random_mask': 'Random Masking',
            'generic_mask': 'Generic Masking'
        }
    )
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
    
    fig, ax = PlotUtils.plot_mic_mltc_classifier_performance(
        performance_df=table_report,
        model_names=[
            "xlm-roberta-large",
            "bert-large-cased",
            "pubmedbert-base-uncased"
        ],
        redaction_strategies=list(strategy_styles.keys()),
        model_bg_colors={
            "xlm-roberta-large": "#cea8bb",
            "bert-large-cased": "#75a190",
            "pubmedbert-base-uncased": "#c9bf89"
        },
        strategy_styles=strategy_styles,
        figsize=(7, 4)
    )

    figure_dir = project_root / "plots" / "DATEXIS" / "med_intent_classification" / "mltc"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_dir / "macro_f1_score_by_model_and_redaction_strategy.png",
        dpi=300,
        bbox_inches="tight"
    )
    """


    ### ECHR Dataset Analysis
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
    """
    
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
    table_report = ReportUtils.get_performance_metrics_summary_table(
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
    table_report = ReportUtils.get_performance_metrics_summary_table(
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
        table_report = ReportUtils.get_performance_metrics_summary_table(
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
            "xlm-roberta-large": "#cea8bb",
            "bert-large-cased": "#75a190",
            "electra-large-discriminator": "#c9bf89"
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
    """
    model_name = 'google--electra-large-discriminator'
    model_name_alias = 'electra-large-discriminator'
    redaction_strategy = 'generic_mask'
    redaction_strategy_alias = 'Generic Masking'
    
    model_performance = ReportUtils.get_performance_metrics_summary_table(
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
    """
    