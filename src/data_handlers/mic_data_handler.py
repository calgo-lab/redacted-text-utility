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
    
    def _get_merged_dataframe_file_path(self) -> Path:
        """
        Returns the path to the merged dataframe parquet file.

        :return: Path to the merged dataframe parquet file.
        """
        return self.data_dir / "processed" / "DATEXIS" / "med_intent_classification" / f"mic_merged.parquet"
    
    def create_merged_dataframe(self) -> Path:
        """
        Merges all raw data files into a single dataframe and saves it as a parquet file.

        :return: Path to the merged dataframe parquet file.
        """
        merged_df_file_path = self._get_merged_dataframe_file_path()
        if merged_df_file_path.exists():
            return merged_df_file_path
        
        raw_dataframes_dict = dict()
        for filename in self.get_available_raw_files():
            split_name = filename.split('-')[0]
            split_df = self.get_dataframe_for_file(filename)
            raw_dataframes_dict[split_name] = split_df
        
        for split, split_df in raw_dataframes_dict.items():
            itemid = f"{split}_" + split_df["__index_level_0__"].astype(str)
            split_df['itemid'] = itemid
            split_df.drop(columns=["__index_level_0__"], inplace=True)
            split_columns = split_df.columns.tolist()
            split_columns.insert(0, split_columns.pop(split_columns.index('itemid')))
            raw_dataframes_dict[split] = split_df[split_columns]
        
        merged_df = pd.concat(raw_dataframes_dict.values(), ignore_index=True)
        merged_df_file_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(merged_df_file_path, 
                             index=False, 
                             engine="pyarrow", 
                             compression="gzip")

        return merged_df_file_path
    
    def get_merged_dataframe(self) -> pd.DataFrame:
        """
        Returns the merged dataframe containing all rows from all data splits.

        :return: pandas dataframe containing the merged data.
        """
        merged_df_path = self._get_merged_dataframe_file_path()
        if not merged_df_path.exists():
            merged_df_path = self.create_merged_dataframe()
        
        return pd.read_parquet(merged_df_path)
    
    def _get_num_tokens_df_file_path(self) -> Path:
        """
        Returns the path to the num_tokens dataframe parquet file.

        :return: Path to the num_tokens dataframe parquet file.
        """
        
        return self.data_dir / "processed" / "DATEXIS" / "med_intent_classification" / "mic_merged_num_tokens.parquet"
    
    def create_num_tokens_df(self, 
                             id_column: str = "itemid", 
                             text_column: str = "text") -> Path:
        """
        Tokenizes the text column using spacy and calculates the 
        number of tokens for each entry in the dataframe and creates 
        a new dataframe with the id column and num_tokens column.
        
        :param id_column: Name of the ID column in the dataframe.
        :param text_column: Name of the text column to tokenize.

        :return: Path to the num_tokens dataframe parquet file.
        """

        num_tokens_df_file_path = self._get_num_tokens_df_file_path()
        if num_tokens_df_file_path.exists():
            return num_tokens_df_file_path

        nlp = spacy.load("en_core_web_sm")
        df = self.get_merged_dataframe()
        
        with tqdm.tqdm(total=len(df), desc="Tokenizing texts and counting tokens") as pbar:
            def count_tokens(text: str) -> int:
                doc = nlp(text)
                token_count = len(doc)
                pbar.update(1)
                return token_count
            
            df['num_tokens'] = df[text_column].apply(count_tokens)
        
        df = df[[id_column, 'num_tokens']]
        df.to_parquet(num_tokens_df_file_path, 
                      index=False, 
                      engine="pyarrow", 
                      compression='gzip')
        
        return num_tokens_df_file_path

    def get_num_tokens_df(self) -> pd.DataFrame:
        """
        Returns the num_tokens dataframe.

        :return: pandas dataframe containing the num_tokens data.
        """
        num_tokens_df_path = self._get_num_tokens_df_file_path()
        if not num_tokens_df_path.exists():
            num_tokens_df_path = self.create_num_tokens_df()
        
        return pd.read_parquet(num_tokens_df_path)
    
    def _get_private_entities_df_file_path(self) -> Path:
        """
        Returns the path to the private entities data parquet file.

        :return: Path to the private entities dataframe parquet file.
        """
        
        return self.data_dir / "processed" / "DATEXIS" / "med_intent_classification" / "mic_merged_pe.parquet"
    
    def get_private_entities_df(self) -> pd.DataFrame:
        """
        Returns the private entities dataframe.

        :return: pandas DataFrame containing the private entities data.
        """
        private_entities_df_path = self._get_private_entities_df_file_path()
        if not private_entities_df_path.exists():
            raise FileNotFoundError(f"Private entities dataframe file not found at: {private_entities_df_path}")
        
        return pd.read_parquet(private_entities_df_path)
    
    def get_private_entity_stats(self, 
                                 input_df: pd.DataFrame, 
                                 id_column: str = "itemid") -> Dict[str, int]:
        """
        Computes statistics about private entities in the input dataframe.

        :param input_df: dataframe containing id_column.
        :param id_column: Name of the ID column in the dataframe.

        :return: Dictionary with statistics about private entities for the input dataframe.
        """

        private_entities_df = self.get_private_entities_df()
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
                                       random_state: int = 2026, 
                                       k: int = 1) -> DatasetDict:
        
        """
        Retrieve the train, dev, and test dataframes for the specified fold.

        :param random_state: Random state for reproducibility.
        :param k: The fold number to retrieve (1-based index).
        :return: A DatasetDict containing the train, dev, and test datasets.
        """
        sample_df = self.get_merged_dataframe()

        pe_df = self.get_private_entities_df()

        itemids_wpe = pe_df[pe_df["pe_count_total"] > 0]["itemid"].unique().tolist()
        itemids_wope = pe_df[pe_df["pe_count_total"] == 0]["itemid"].unique().tolist()

        sample_df_wpe = sample_df[sample_df["itemid"].isin(itemids_wpe)].copy()
        sample_df_wope = sample_df[sample_df["itemid"].isin(itemids_wope)].copy()

        sample_df_wpe = sample_df_wpe.reset_index(drop=True)
        sample_df_wope = sample_df_wope.reset_index(drop=True)

        fold_tuples = list()
        splits_wpe = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df_wpe.index.to_numpy()))
        splits_wope = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df_wope.index.to_numpy()))

        train_dev_test_k_folds = self.get_train_dev_test_folds()
        for index, fold in enumerate(train_dev_test_k_folds):
            
            train_indices_wpe = list()
            train_indices_wope = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices_wpe += list(splits_wpe[fold_train_index][1])
                train_indices_wope += list(splits_wope[fold_train_index][1])
            
            dev_indices_wpe = list()
            dev_indices_wope = list()
            fold_dev_indices = fold[2]
            for fold_dev_index in fold_dev_indices:
                dev_indices_wpe += list(splits_wpe[fold_dev_index][1])
                dev_indices_wope += list(splits_wope[fold_dev_index][1])
            
            test_indices = list()
            fold_test_indices = fold[3]
            for fold_test_index in fold_test_indices:
                test_indices += list(splits_wpe[fold_test_index][1])
            
            fold_tuples.append((
                index + 1,
                sample_df_wpe[sample_df_wpe.index.isin(train_indices_wpe)].itemid.tolist() + sample_df_wope[sample_df_wope.index.isin(train_indices_wope)].itemid.tolist(),
                sample_df_wpe[sample_df_wpe.index.isin(dev_indices_wpe)].itemid.tolist() + sample_df_wope[sample_df_wope.index.isin(dev_indices_wope)].itemid.tolist(),
                sample_df_wpe[sample_df_wpe.index.isin(test_indices)].itemid.tolist()
            ))

        kth_tuple = fold_tuples[k-1]

        split_order = {
            "train": 0, 
            "validation": 1, 
            "test": 2
        }
        
        train_df = sample_df[sample_df.itemid.isin(kth_tuple[1])].copy()
        train_df = train_df.sort_values(
            by="itemid",
            key=lambda s: s.map(
                lambda x: (
                    split_order[x.split("_")[0]],
                    int(x.split("_")[1])
                )
            )
        ).reset_index(drop=True)
        train_ds = Dataset.from_pandas(train_df)

        dev_df = sample_df[sample_df.itemid.isin(kth_tuple[2])].copy()
        dev_df = dev_df.sort_values(
            by="itemid",
            key=lambda s: s.map(
                lambda x: (
                    split_order[x.split("_")[0]],
                    int(x.split("_")[1])
                )
            )
        ).reset_index(drop=True)
        dev_ds = Dataset.from_pandas(dev_df)

        test_df = sample_df[sample_df.itemid.isin(kth_tuple[3])].copy()
        test_df = test_df.sort_values(
            by="itemid",
            key=lambda s: s.map(
                lambda x: (
                    split_order[x.split("_")[0]],
                    int(x.split("_")[1])
                )
            )
        ).reset_index(drop=True)
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
                       id_column: str = "itemid") -> Dict[str, str]:
        """
        Given a DatasetDict with 'train', 'dev', 'test' splits,
        returns a dict with total tokens, entities, and per-label entity counts for each split.

        :param fold_datasetdict: The DatasetDict containing 'train', 'dev', 'test' datasets.
        :param id_column: Name of the ID column in the DataFrame.
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
        
        train_intent_counts: Dict[str, int] = dict()
        for intents in train_df["intents"]:
            for intent in intents:
                train_intent_counts[intent] = train_intent_counts.get(intent, 0) + 1
        train_intent_counts = dict(sorted(train_intent_counts.items()))

        dev_intent_counts: Dict[str, int] = dict()
        for intents in dev_df["intents"]:
            for intent in intents:
                dev_intent_counts[intent] = dev_intent_counts.get(intent, 0) + 1
        dev_intent_counts = dict(sorted(dev_intent_counts.items()))

        test_intent_counts: Dict[str, int] = dict()
        for intents in test_df["intents"]:
            for intent in intents:
                test_intent_counts[intent] = test_intent_counts.get(intent, 0) + 1
        test_intent_counts = dict(sorted(test_intent_counts.items()))

        stats["class_counts"] = {
            "train": train_intent_counts,
            "dev": dev_intent_counts,
            "test": test_intent_counts
        }

        num_tokens_df = self.get_num_tokens_df()
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

        pe_df = self.get_private_entities_df()
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
                    "min": int(round(pe_df_train[count_col].min(), 0)),
                    "max": int(round(pe_df_train[count_col].max(), 0))
                },
                "dev": {
                    "total": int(round(pe_df_dev[count_col].sum(), 0)),
                    "min": int(round(pe_df_dev[count_col].min(), 0)),
                    "max": int(round(pe_df_dev[count_col].max(), 0))
                },
                "test": {
                    "total": int(round(pe_df_test[count_col].sum(), 0)),
                    "min": int(round(pe_df_test[count_col].min(), 0)),
                    "max": int(round(pe_df_test[count_col].max(), 0))
                }
            }

        return json.loads(json.dumps(stats, default=lambda x: x.item()))