from pathlib import Path

from tqdm import tqdm

from core.logging import get_logger
from infrastructure.services.model_service import ModelService
from infrastructure.services.model_service_impl import ModelServiceImpl

import json

import pandas as pd


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