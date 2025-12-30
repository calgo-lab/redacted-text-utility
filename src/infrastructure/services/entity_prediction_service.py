from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from core.logging import get_logger
from infrastructure.services.model_service import ModelService
from infrastructure.services.model_service_impl import ModelServiceImpl

import os
import json
import threading

import pandas as pd


_worker_mim = None

def _init_worker(entity_set_id: str, model_id: str):
    """
    Initializer: runs ONCE per process.
    """
    global _worker_mim
    model_service = ModelServiceImpl()
    _worker_mim = model_service.get_model_inference_maker(entity_set_id, model_id)


def _infer_single(idx: int, text: str):
    """
    Worker function: runs in each process.
    """
    global _worker_mim
    result = _worker_mim.infer(text)
    return idx, json.dumps(result), len(result) if isinstance(result, list) else 0


class EntityPredictionService:
    """
    High-level helper service for entity predictions using NER models.
    """

    def __init__(self):
        """
        Initializes the EntityPredictionService with a ModelService instance.
        """
        self.logger = get_logger(__name__)
        self._model_service: ModelService = ModelServiceImpl()
    
    def collect_named_entities_for_dataframe(self, 
                                             entity_set_id: str, 
                                             model_id: str, 
                                             source_df: pd.DataFrame, 
                                             source_column: str, 
                                             target_column: str | None, 
                                             target_df_export_path: Path) -> Path | None:
        
        """
        Collects named entities for each text entry in the specified column of the dataframe
        using the specified NER model and exports the results to a new column in the dataframe.

        :param entity_set_id: The ID of the entity set to use for predictions.
        :param model_id: The ID of the NER model to use for predictions.
        :param source_df: The pandas DataFrame containing the text data.
        :param source_column: The name of the column in the dataframe containing text to analyze.
        :param target_column: The name of the column in the dataframe where predictions will be stored.
        :param target_df_export_path: The path where the updated dataframe with predictions will be saved.
        :return: The path to the exported dataframe with named entity predictions, or None if export fails.
        """

        mim = self._model_service.get_model_inference_maker(entity_set_id, model_id)
        if target_column is None:
            target_column = f"{source_column}_ne_{entity_set_id}_{model_id}"

        source_df = source_df.copy()
        total_entities = [0]
    
        def infer_and_count(text):
            result = mim.infer(text)
            entity_count = len(result) if isinstance(result, list) else 0
            total_entities[0] += entity_count
            return json.dumps(result)

        with tqdm(total=len(source_df), desc=f"Collecting named entities") as pbar:
            source_df[target_column] = source_df[source_column].apply(
                lambda text: (
                    pbar.update(1), 
                    infer_and_count(text), 
                    pbar.set_postfix({
                        "total_entities": total_entities[0]
                    })
                )[1]
            )

        try:
            source_df.to_parquet(target_df_export_path, index=False)
            return target_df_export_path
        except Exception as e:
            print(f"Failed to export dataframe to {target_df_export_path}: {e}")
            return None
    
    def collect_named_entities_for_dataframe_parallel(self,
                                                      entity_set_id: str,
                                                      model_id: str,
                                                      source_df: pd.DataFrame,
                                                      source_column: str,
                                                      target_column: str | None,
                                                      target_df_export_path: Path,
                                                      max_workers: int | None = None) -> Path | None:
        """
        Collects named entities for each text entry in the specified column of the dataframe
        using the specified NER model in parallel and exports the results to a new column in 
        the dataframe.

        :param entity_set_id: The ID of the entity set to use for predictions.
        :param model_id: The ID of the NER model to use for predictions.
        :param source_df: The pandas DataFrame containing the text data.
        :param source_column: The name of the column in the dataframe containing text to analyze.
        :param target_column: The name of the column in the dataframe where predictions will be stored.
        :param target_df_export_path: The path where the updated dataframe with predictions will be saved.
        :param max_workers: The maximum number of worker threads to use for parallel processing.
        :return: The path to the exported dataframe with named entity predictions, or None if export fails.
        """

        if target_column is None:
            target_column = f"{source_column}_ne_{entity_set_id}_{model_id}"

        source_df = source_df.copy()
        source_df[target_column] = None

        max_workers = max_workers or max(1, (os.cpu_count() or 4) // 2)
        self.logger.info(f"Using max_workers={max_workers} (ProcessPoolExecutor)")

        texts = list(source_df[source_column].items())
        total_entities = 0

        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=_init_worker,
                                 initargs=(entity_set_id, model_id)) as executor:

            futures = {
                executor.submit(_infer_single, idx, text): idx
                for idx, text in texts
            }

            with tqdm(total=len(futures), desc="Collecting named entities") as pbar:
                for future in as_completed(futures):
                    idx, result_json, entity_count = future.result()
                    source_df.at[idx, target_column] = result_json
                    total_entities += entity_count

                    pbar.update(1)
                    pbar.set_postfix({"total_entities": total_entities})

        try:
            target_df_export_path.parent.mkdir(parents=True, exist_ok=True)
            source_df.to_parquet(target_df_export_path, index=False)
            return target_df_export_path
        except Exception as e:
            self.logger.exception(
                f"Failed to export dataframe to {target_df_export_path}"
            )
            return None