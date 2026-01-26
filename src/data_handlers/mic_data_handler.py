from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download

from core.logging import get_logger

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