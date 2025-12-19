from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.model_selection import KFold

from core.logging import get_logger

import numpy as np
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
    
    def get_train_dev_test_datasetdict(self, 
                                       random_state: int = 2025,
                                       k: int = 1) -> DatasetDict:
        
        """
        Retrieve the train, dev, and test dataframes for the specified fold.

        :param random_state: Random state for reproducibility.
        :param k: The fold number to retrieve (1-based index).
        :return: A DatasetDict containing the train, dev, and test datasets.
        """
        sample_df = self.get_dataframe_for_file("ECHR_Dataset.parquet")

        sample_df_0 = sample_df[sample_df['binary_judgement'] == 0].reset_index(drop=True)
        sample_df_1 = sample_df[sample_df['binary_judgement'] == 1].reset_index(drop=True)

        fold_tuples = list()
        splits_0 = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df_0.index.to_numpy()))
        splits_1 = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df_1.index.to_numpy()))
        
        train_dev_test_k_folds = self.get_train_dev_test_folds()
        for index, fold in enumerate(train_dev_test_k_folds):
            train_indices_0 = list()
            train_indices_1 = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices_0 += list(splits_0[fold_train_index][1])
                train_indices_1 += list(splits_1[fold_train_index][1])
            dev_indices_0 = list()
            dev_indices_1 = list()
            fold_dev_indices = fold[2]
            for fold_dev_index in fold_dev_indices:
                dev_indices_0 += list(splits_0[fold_dev_index][1])
                dev_indices_1 += list(splits_1[fold_dev_index][1])
            test_indices_0 = list()
            test_indices_1 = list()
            fold_test_indices = fold[3]
            for fold_test_index in fold_test_indices:
                test_indices_0 += list(splits_0[fold_test_index][1])
                test_indices_1 += list(splits_1[fold_test_index][1])
            
            fold_tuples.append((
                index + 1,
                sample_df_0[sample_df_0.index.isin(train_indices_0)].itemid.tolist() + sample_df_1[sample_df_1.index.isin(train_indices_1)].itemid.tolist(),
                sample_df_0[sample_df_0.index.isin(dev_indices_0)].itemid.tolist() + sample_df_1[sample_df_1.index.isin(dev_indices_1)].itemid.tolist(),
                sample_df_0[sample_df_0.index.isin(test_indices_0)].itemid.tolist() + sample_df_1[sample_df_1.index.isin(test_indices_1)].itemid.tolist()
            ))
        
        kth_tuple = fold_tuples[k-1]

        train_df = sample_df[sample_df.itemid.isin(kth_tuple[1])].copy()
        train_df["itemid_num"] = train_df["itemid"].str.split("-").str[1].astype(int)
        train_df = train_df.sort_values(by="itemid_num").reset_index(drop=True)
        train_df = train_df.drop(columns=["itemid_num"])
        train_ds = Dataset.from_pandas(train_df)

        dev_df = sample_df[sample_df.itemid.isin(kth_tuple[2])].copy()
        dev_df["itemid_num"] = dev_df["itemid"].str.split("-").str[1].astype(int)
        dev_df = dev_df.sort_values(by="itemid_num").reset_index(drop=True)
        dev_df = dev_df.drop(columns=["itemid_num"])
        dev_ds = Dataset.from_pandas(dev_df)

        test_df = sample_df[sample_df.itemid.isin(kth_tuple[3])].copy()
        test_df["itemid_num"] = test_df["itemid"].str.split("-").str[1].astype(int)
        test_df = test_df.sort_values(by="itemid_num").reset_index(drop=True)
        test_df = test_df.drop(columns=["itemid_num"])
        test_ds = Dataset.from_pandas(test_df)

        return DatasetDict({
            "train": train_ds, 
            "dev": dev_ds, 
            "test": test_ds
        })
    
    @staticmethod
    def get_train_dev_test_folds(n_fold: int = 5, 
                                 train_percent: float = 0.6, 
                                 dev_percent: float = 0.2) -> List[Tuple]:
        """
        Generates train, dev, and test fold indices for k-fold cross-validation.
        
        :param n_fold: Total number of folds.
        :param train_percent: Percentage of data to be used for training.
        :param dev_percent: Percentage of data to be used for development/validation.
        :return: List of tuples containing fold number, train indices, dev indices, and test indices.
        """
        fold_tuples = list()
        indices = list(range(n_fold))
        train_start = 0
        train_end = int(round(n_fold * train_percent))
        dev_start = train_end
        dev_end = int(round(n_fold * (train_percent + dev_percent)))
        test_start = dev_end
        test_end = n_fold
        for index in indices:
            rolled_indices = np.roll(indices, -index)
            train_indices = list(rolled_indices[train_start: train_end])
            dev_indices = list(rolled_indices[dev_start: dev_end])
            test_indices = list(rolled_indices[test_start: test_end])
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices,
                test_indices
            ))
        return fold_tuples