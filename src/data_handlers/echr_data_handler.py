from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download

from core.logging import get_logger

import pandas as pd


class EchrDataHandler:
    """
    Data handler for the ECHR dataset, a collection of 11.5K court cases 
    extracted from the public database of the European Court of Human Rights 
    and further annotated by human experts.
    
    https://huggingface.co/datasets/glnmario/ECHR
    https://www.aclweb.org/anthology/P19-1424
    https://archive.org/details/ECHR-ACL2019
    """

    def __init__(self, project_root: Path, data_dir: Path = None):
        """
        Initializes the EchrDataHandler.

        :param project_root: Path to the root of the project.
        :param data_dir: Optional path to the data directory. If not provided, defaults to 'data' folder in project root.
        """
        self.logger = get_logger(__name__)
        self.project_root = project_root
        if data_dir:
            self.data_dir: Path = data_dir
        else:
            self.data_dir: Path = self.project_root / "data"

        self.hf_repo_id: str = "glnmario/ECHR"
        self.hf_files: List[str] = [
            "ECHR_Dataset.csv"
        ]
        self.raw_files_dict: Dict[str, Path] = dict()
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Loads the ECHR dataset.
        First checks if the data files exist locally; if not, downloads them from Hugging Face Hub.

        :return: None
        """
        for filename in self.hf_files:
            local_file_path = self.data_dir / "raw" / "glnmario" / "ECHR" / filename
            parquet_file_path = local_file_path.with_suffix('.parquet')
            if parquet_file_path.exists():
                self.raw_files_dict[parquet_file_path.name] = parquet_file_path
                continue
            else:
                if local_file_path.exists():
                    self.logger.info(f"Found existing local file: {local_file_path}")
                else:
                    self.logger.info(f"File {filename} not found locally. Downloading from Hugging Face Hub...")
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename=f"{filename}",
                        repo_type="dataset",
                        local_dir=local_file_path.parent
                    )
                    self.logger.info(f"Downloaded {filename} to {local_file_path}")
                self._convert_to_parquet(local_file_path)
                self.raw_files_dict[parquet_file_path.name] = parquet_file_path
    
    def _convert_to_parquet(self, csv_file_path: Path) -> Path:
        """
        Converts a CSV file to Parquet format.

        :param csv_file_path: Path to the CSV file.
        :return: Path to the converted Parquet file.
        """
        parquet_file_path = csv_file_path.with_suffix('.parquet')
        df = pd.read_csv(csv_file_path)
        df.to_parquet(parquet_file_path, index=False, engine="pyarrow", compression='gzip')
        self.logger.info(f"Converted {csv_file_path} to {parquet_file_path}")

        # Delete the original CSV file
        csv_file_path.unlink()
        self.logger.info(f"Deleted original CSV file: {csv_file_path}")

        return parquet_file_path
    
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