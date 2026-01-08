from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.model_selection import KFold

from core.logging import get_logger

import json

import numpy as np
import pandas as pd
import spacy
import tqdm


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
    
    def _get_num_tokens_df_file_path(self, filename: str) -> Path:
        """
        Returns the path to the num_tokens DataFrame parquet file.

        :param filename: Name of the data file.

        :return: Path to the num_tokens DataFrame parquet file.
        """
        if filename.endswith('.parquet'):
            filename = filename[:-8]
        
        return self.data_dir / "processed" / "glnmario" / "ECHR" / f"{filename}_num_tokens.parquet"
    
    def get_num_tokens_df(self, filename: str = "ECHR_Dataset.parquet") -> pd.DataFrame:
        """
        Returns the num_tokens DataFrame for the specified data file.

        :param filename: Name of the data file, default is "ECHR_Dataset.parquet".

        :return: pandas DataFrame containing the num_tokens data.
        """
        num_tokens_df_path = self._get_num_tokens_df_file_path(filename)
        if not num_tokens_df_path.exists():
            num_tokens_df_path = self.create_num_tokens_df_for_data_file(filename)
        
        return pd.read_parquet(num_tokens_df_path)
    
    def create_num_tokens_df_for_data_file(self, 
                                           filename: str, 
                                           id_column: str = "itemid", 
                                           text_column: str = "text") -> Path:
        """
        Tokenizes the text column using spacy and calculates the 
        number of tokens for each entry in the DataFrame and creates 
        a new DataFrame with the id column and num_tokens column.
        
        :param filename: Name of the data file to process.
        :param id_column: Name of the ID column in the DataFrame.
        :param text_column: Name of the text column to tokenize.

        :return: Path to the num_tokens dataframe parquet file.
        """

        num_tokens_df_file_path = self._get_num_tokens_df_file_path(filename)
        if num_tokens_df_file_path.exists():
            return num_tokens_df_file_path

        nlp = spacy.load("en_core_web_sm")
        df = self.get_dataframe_for_file(filename)
        
        with tqdm.tqdm(total=len(df), desc="Tokenizing texts and counting tokens") as pbar:
            def count_tokens(text: str) -> int:
                doc = nlp(text)
                token_count = len(doc)
                pbar.update(1)
                return token_count
            
            df['num_tokens'] = df[text_column].apply(count_tokens)
        
        df = df[[id_column, 'num_tokens']]
        df.to_parquet(num_tokens_df_file_path, index=False, engine="pyarrow", compression='gzip')
        return num_tokens_df_file_path
    
    def _get_private_entities_df_file_path(self, filename: str) -> Path:
        """
        Returns the path to the private entities DataFrame parquet file.

        :param filename: Name of the data file.

        :return: Path to the private entities DataFrame parquet file.
        """
        if filename.endswith('.parquet'):
            filename = filename[:-8]
        
        return self.data_dir / "processed" / "glnmario" / "ECHR" / f"{filename}_pe.parquet"
    
    def get_private_entities_df(self, filename: str) -> pd.DataFrame:
        """
        Returns the private entities DataFrame for the specified data file.

        :param filename: Name of the data file.

        :return: pandas DataFrame containing the private entities data.
        """
        private_entities_df_path = self._get_private_entities_df_file_path(filename)
        if not private_entities_df_path.exists():
            raise FileNotFoundError(f"Private entities DataFrame file not found at: {private_entities_df_path}")
        
        return pd.read_parquet(private_entities_df_path)
    
    def get_private_entity_stats(self, 
                                 input_df: pd.DataFrame, 
                                 filename: str = "ECHR_Dataset.parquet", 
                                 id_column: str = "itemid") -> Dict[str, int]:
        """
        Computes statistics about private entities in the input DataFrame.

        :param input_df: DataFrame containing id_column.
        :param filename: Name of the data file, default is "ECHR_Dataset.parquet".
        :param id_column: Name of the ID column in the DataFrame.

        :return: Dictionary with statistics about private entities for the input DataFrame.
        """

        private_entities_df = self.get_private_entities_df(filename)
        count_columns = private_entities_df.columns.tolist()
        count_columns = [
            count_col 
            for count_col in count_columns 
            if count_col.startswith('pe_count_')
        ]
        
        private_entities_df_filtered = private_entities_df[
            private_entities_df[id_column].isin(input_df[id_column])
        ]

        stats: Dict[str, int] = dict()
        for _, row in private_entities_df_filtered.iterrows():
            for count_col in count_columns:
                count = row[count_col]
                if count_col not in stats:
                    stats[count_col] = 0
                stats[count_col] += count
        
        return stats
    
    def get_train_dev_test_datasetdict(self, 
                                       filename: str = "ECHR_Dataset.parquet",
                                       random_state: int = 2025, 
                                       k: int = 1) -> DatasetDict:
        
        """
        Retrieve the train, dev, and test dataframes for the specified fold.

        :param random_state: Random state for reproducibility.
        :param k: The fold number to retrieve (1-based index).
        :return: A DatasetDict containing the train, dev, and test datasets.
        """
        sample_df = self.get_dataframe_for_file(filename)

        num_tokens_df = self.get_num_tokens_df(filename)
        num_tokens_df_filtered = num_tokens_df[
            (num_tokens_df["num_tokens"] >= 512) & (num_tokens_df["num_tokens"] <= 5120)
        ]

        sample_df_filtered = sample_df[
            sample_df["itemid"].isin(num_tokens_df_filtered["itemid"])
        ].reset_index(drop=True)

        sample_df_0 = sample_df_filtered[sample_df_filtered['binary_judgement'] == 0].reset_index(drop=True)
        sample_df_1 = sample_df_filtered[sample_df_filtered['binary_judgement'] == 1].reset_index(drop=True)

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

        train_df = sample_df_filtered[sample_df_filtered.itemid.isin(kth_tuple[1])].copy()
        train_df["itemid_num"] = train_df["itemid"].str.split("-").str[1].astype(int)
        train_df = train_df.sort_values(by="itemid_num").reset_index(drop=True)
        train_df = train_df.drop(columns=["itemid_num"])
        train_ds = Dataset.from_pandas(train_df)

        dev_df = sample_df_filtered[sample_df_filtered.itemid.isin(kth_tuple[2])].copy()
        dev_df["itemid_num"] = dev_df["itemid"].str.split("-").str[1].astype(int)
        dev_df = dev_df.sort_values(by="itemid_num").reset_index(drop=True)
        dev_df = dev_df.drop(columns=["itemid_num"])
        dev_ds = Dataset.from_pandas(dev_df)

        test_df = sample_df_filtered[sample_df_filtered.itemid.isin(kth_tuple[3])].copy()
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
    
    def get_fold_stats(self,
                       fold_datasetdict: DatasetDict,
                       filename: str = "ECHR_Dataset.parquet",
                       id_column: str = "itemid") -> Dict[str, str]:
        """
        Given a DatasetDict with 'train', 'dev', 'test' splits,
        returns a dict with total tokens, entities, and per-label entity counts for each split.

        :param fold_datasetdict: The DatasetDict containing 'train', 'dev', 'test' datasets.
        :return: A dictionary with stats as keys and formatted strings as values
        """
        train_df = fold_datasetdict["train"].to_pandas()
        dev_df = fold_datasetdict["dev"].to_pandas()
        test_df = fold_datasetdict["test"].to_pandas()

        stats: Dict[str, str] = dict()
        stats["total_documents"] = {
            "train": len(train_df[id_column].unique()),
            "dev": len(dev_df[id_column].unique()),
            "test": len(test_df[id_column].unique())
        }

        stats["class_counts"] = {
            "train": {
                "binary_judgement_0": len(train_df[train_df['binary_judgement'] == 0]),
                "binary_judgement_1": len(train_df[train_df['binary_judgement'] == 1]),
            },
            "dev": {
                "binary_judgement_0": len(dev_df[dev_df['binary_judgement'] == 0]),
                "binary_judgement_1": len(dev_df[dev_df['binary_judgement'] == 1]),
            },
            "test": {
                "binary_judgement_0": len(test_df[test_df['binary_judgement'] == 0]),
                "binary_judgement_1": len(test_df[test_df['binary_judgement'] == 1]),
            }
        }

        num_tokens_df = self.get_num_tokens_df(filename)
        num_tokens_df_train = num_tokens_df[num_tokens_df[id_column].isin(train_df[id_column])]
        num_tokens_df_dev = num_tokens_df[num_tokens_df[id_column].isin(dev_df[id_column])]
        num_tokens_df_test = num_tokens_df[num_tokens_df[id_column].isin(test_df[id_column])]

        stats["token_stats"] = {
            "train": {
                "total": num_tokens_df_train['num_tokens'].sum(),
                "mean": int(round(num_tokens_df_train['num_tokens'].mean(), 0)),
                "std": int(round(num_tokens_df_train['num_tokens'].std(), 0)),
                "min": int(round(num_tokens_df_train['num_tokens'].min(), 0)),
                "25p": int(round(num_tokens_df_train['num_tokens'].quantile(0.25), 0)),
                "median": int(round(num_tokens_df_train['num_tokens'].quantile(0.50), 0)),
                "75p": int(round(num_tokens_df_train['num_tokens'].quantile(0.75), 0)),
                "max": int(round(num_tokens_df_train['num_tokens'].max(), 0))
            },
            "dev": {
                "total": int(round(num_tokens_df_dev['num_tokens'].sum(), 0)),
                "mean": int(round(num_tokens_df_dev['num_tokens'].mean(), 0)),
                "std": int(round(num_tokens_df_dev['num_tokens'].std(), 0)),
                "min": int(round(num_tokens_df_dev['num_tokens'].min(), 0)),
                "25p": int(round(num_tokens_df_dev['num_tokens'].quantile(0.25), 0)),
                "median": int(round(num_tokens_df_dev['num_tokens'].quantile(0.50), 0)),
                "75p": int(round(num_tokens_df_dev['num_tokens'].quantile(0.75), 0)),
                "max": int(round(num_tokens_df_dev['num_tokens'].max(), 0))
            },
            "test": {
                "total": int(round(num_tokens_df_test['num_tokens'].sum(), 0)),
                "mean": int(round(num_tokens_df_test['num_tokens'].mean(), 0)),
                "std": int(round(num_tokens_df_test['num_tokens'].std(), 0)),
                "min": int(round(num_tokens_df_test['num_tokens'].min(), 0)),
                "25p": int(round(num_tokens_df_test['num_tokens'].quantile(0.25), 0)),
                "median": int(round(num_tokens_df_test['num_tokens'].quantile(0.50), 0)),
                "75p": int(round(num_tokens_df_test['num_tokens'].quantile(0.75), 0)),
                "max": int(round(num_tokens_df_test['num_tokens'].max(), 0))
            }
        }

        pe_df = self.get_private_entities_df(filename)
        pe_df_train = pe_df[pe_df[id_column].isin(train_df[id_column])]
        pe_df_dev = pe_df[pe_df[id_column].isin(dev_df[id_column])]
        pe_df_test = pe_df[pe_df[id_column].isin(test_df[id_column])]

        count_columns = pe_df.columns.tolist()
        count_columns = [
            count_col 
            for count_col in count_columns 
            if count_col.startswith('pe_count_')
        ]

        stats["private_entity_stats"] = dict()
        for count_col in count_columns:
            stats["private_entity_stats"][count_col] = {
                "train": {
                    "total": int(round(pe_df_train[count_col].sum(), 0)),
                    "mean": int(round(pe_df_train[count_col].mean(), 0)),
                    "std": int(round(pe_df_train[count_col].std(), 0)),
                    "min": int(round(pe_df_train[count_col].min(), 0)),
                    "25p": int(round(pe_df_train[count_col].quantile(0.25), 0)),
                    "median": int(round(pe_df_train[count_col].quantile(0.50), 0)),
                    "75p": int(round(pe_df_train[count_col].quantile(0.75), 0)),
                    "max": int(round(pe_df_train[count_col].max(), 0))
                },
                "dev": {
                    "total": int(round(pe_df_dev[count_col].sum(), 0)),
                    "mean": int(round(pe_df_dev[count_col].mean(), 0)),
                    "std": int(round(pe_df_dev[count_col].std(), 0)),
                    "min": int(round(pe_df_dev[count_col].min(), 0)),
                    "25p": int(round(pe_df_dev[count_col].quantile(0.25), 0)),
                    "median": int(round(pe_df_dev[count_col].quantile(0.50), 0)),
                    "75p": int(round(pe_df_dev[count_col].quantile(0.75), 0)),
                    "max": int(round(pe_df_dev[count_col].max(), 0))
                },
                "test": {
                    "total": int(round(pe_df_test[count_col].sum(), 0)),
                    "mean": int(round(pe_df_test[count_col].mean(), 0)),
                    "std": int(round(pe_df_test[count_col].std(), 0)),
                    "min": int(round(pe_df_test[count_col].min(), 0)),
                    "25p": int(round(pe_df_test[count_col].quantile(0.25), 0)),
                    "median": int(round(pe_df_test[count_col].quantile(0.50), 0)),
                    "75p": int(round(pe_df_test[count_col].quantile(0.75), 0)),
                    "max": int(round(pe_df_test[count_col].max(), 0))
                }
            }

        return json.loads(json.dumps(stats, default=lambda x: x.item()))