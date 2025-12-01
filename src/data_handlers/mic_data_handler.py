from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download

from core.logging import get_logger

import pandas as pd


class MicDataHandler:
    """
    Data handler for Medical Intent Classification (MIC) dataset from DATEXIS.
    """

    def __init__(self, project_root: Path, data_dir: Path = None):
        """
        Initializes the MicDataHandler.

        :param project_root: Path to the root of the project.
        :param data_dir: Optional path to the data directory. If not provided, defaults to 'data' folder in project root.
        """
        self.logger = get_logger(__name__)
        self.project_root = project_root
        if data_dir:
            self.data_dir: Path = data_dir
        else:
            self.data_dir: Path = self.project_root / "data"

        self.hf_repo_id: str = "DATEXIS/med_intent_classification"
        self.hf_files: List[str] = [
            "train-00000-of-00001.parquet",
            "validation-00000-of-00001.parquet",
            "test-00000-of-00001.parquet"
        ]
        self.raw_files_dict: Dict[str, Path] = dict()
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Loads the MIC dataset.
        First checks if the data files exist locally; if not, downloads them from Hugging Face Hub.

        :return: None
        """
        for filename in self.hf_files:
            local_file_path = self.data_dir / "raw" / "DATEXIS" / "med_intent_classification" / "data" / filename
            if not local_file_path.exists():
                self.logger.info(f"File {filename} not found locally. Downloading from Hugging Face Hub...")
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=f"data/{filename}",
                    repo_type="dataset",
                    local_dir=local_file_path.parent.parent
                )
                self.logger.info(f"Downloaded {filename} to {local_file_path}")
            self.raw_files_dict[filename] = local_file_path
    
    def get_available_raw_files(self) -> List[str]:
        """
        Returns a list of available raw data files.

        :return: List of filenames.
        """
        return list(self.raw_files_dict.keys())
    
    def get_dataframe_for_file(self, filename: str) -> pd.DataFrame:
        """
        Returns a pandas DataFrame for the specified file.
        
        :param filename: Name of the file to load.
        :return: pandas DataFrame containing the data.
        """
        if filename not in self.raw_files_dict:
            raise ValueError(f"File {filename} not found in the available files.")
        
        file_path = self.raw_files_dict[filename]
        self.logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        return df