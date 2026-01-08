from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

import json
import random
import re
import string

import pandas as pd


class TokenTreatmentUtils:

    @staticmethod
    def _is_date_token_skippable(token: str) -> bool:
        """
        Determines if a DATE token should be skipped based on specific criteria.
        
        :param token: The DATE token to evaluate.
        :return: True if the token should be skipped, False otherwise.
        """
        skip_token = False
        if not any(char.isdigit() for char in token):
            skip_token = True
        if (
            re.search(r'\bto\b', token) or 
            re.search(r'\bunder\b', token) or 
            re.search(r'\babout\b', token) or 
            re.search(r'\bnext\b', token) or 
            re.search(r'\bthe\b', token) or
            re.search(r'\b300\b', token)
        ):
            skip_token = True
        if (
            'twenty eighteen' in token.lower() or
            'ninety seven' in token.lower() or 
            'ninety nine' in token.lower()
        ):
            skip_token = False

        return skip_token
    
    @staticmethod
    def _get_unusual_person_tokens() -> set:
        """
        Provides a set of known unusual PERSON tokens to exclude.
        
        :return: A set of PERSON tokens to exclude.
        """
        return {
            'the covid', # [train][row: 1187]
            'gotcha', # [train][row: 2428]
            'hawkins-kennedy', # [train][row: 3537]
            'murphy', # [train][row: 3872]
            'gon' # [train][row: 3878]
        }
    
    @staticmethod
    def filter_named_entities(named_entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters named entities by excluding non-private and unusual tokens.

        :param named_entities: The list of named entities to filter.
        :return: A tuple containing the filtered list and the excluded list.
        """
        private_entities: List[Dict[str, Any]] = list()
        excluded_private_entities: List[Dict[str, Any]] = list()
        
        private_entity_labels: Set[str] = {'DATE', 'GPE', 'ORG', 'PERSON'}
        for ne in named_entities:
            if ne['label'] in private_entity_labels:
                if ne['label'] == 'DATE':
                    token = ne['token']
                    skip_token = TokenTreatmentUtils._is_date_token_skippable(token)
                    if not skip_token:
                        private_entities.append(ne)
                    else:
                        excluded_private_entities.append(ne)
                elif ne['label'] == 'PERSON':
                    unusual_tokens = TokenTreatmentUtils._get_unusual_person_tokens()
                    if ne['token'] not in unusual_tokens:
                        private_entities.append(ne)
                    else:
                        excluded_private_entities.append(ne)
                else:
                    private_entities.append(ne)
        
        return private_entities, excluded_private_entities
    
    @staticmethod
    def filter_named_entities_for_dataframe(ne_df: pd.DataFrame,
                                            ne_column: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters named entities by excluding non-private and unusual tokens for a DataFrame.

        :param ne_df: The DataFrame containing named entities to filter.
        :param ne_column: The column name in the DataFrame that contains the named entities as list of dictionaries.
        :return: A tuple containing the filtered list and the excluded list.
        """
        private_entities: List[Dict[str, Any]] = list()
        excluded_private_entities: List[Dict[str, Any]] = list()
        
        for idx, row in ne_df.iterrows():
            ne_list = json.loads(row[ne_column])
            pe_list, epe_list = TokenTreatmentUtils.filter_named_entities(ne_list)
            [pe.update({"row_idx": idx}) for pe in pe_list]
            [epe.update({"row_idx": idx}) for epe in epe_list]
            private_entities.extend(pe_list)
            excluded_private_entities.extend(epe_list)

        return private_entities, excluded_private_entities
    
    @staticmethod
    def generate_random_string(length: int) -> str:
        """
        Generates a random alphanumeric string of the specified length.
        
        :param length: The length of the random string to generate.
        :return: A random alphanumeric string.
        """
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
        
    
    @staticmethod
    def redact_private_entity_tokens_in_text(input_text: str, 
                                             private_entities: List[Dict[str, Any]], 
                                             replacement_strategy: str = "semantic_label_mask") -> str:
        """
        Redacts private entity tokens in the given text based on the specified replacement strategy.

        :param input_text: The original text.
        :param private_entities: A list of private entities with their details.
        :param replacement_strategy: The strategy for replacing tokens. 
               Options are - 
               (1) "semantic_label_mask" or (e.g. "maria martin" -> [PERSON])
               (2) "random_mask" or (e.g. "maria martin" [12] -> Xch6YTvb8mQz)
               (3) "generic_mask" or (e.g. "maria martin" -> XXXX)
        :return: The redacted text.
        """
        redacted_text: str = input_text
        offset: int = 0
        for entity in sorted(private_entities, key=lambda x: x['start']):
            
            token: str = entity['token']
            label: str = entity['label']
            start: int = entity['start']
            end: int = entity['end']
            
            if replacement_strategy == "semantic_label_mask":
                replacement_token = f"[{label}]"
            elif replacement_strategy == "random_mask":
                replacement_token = TokenTreatmentUtils.generate_random_string(len(token))
            elif replacement_strategy == "generic_mask":
                replacement_token = 'XXXX'
            else:
                raise ValueError(f"Unknown replacement strategy: {replacement_strategy}")
            
            redacted_text = redacted_text[:(start+offset)] + replacement_token + redacted_text[(end+offset):]
            offset += len(replacement_token) - (end - start)
        
        return redacted_text
    
    @staticmethod
    def redact_private_entity_tokens_in_text_for_dataframe(ne_df: pd.DataFrame,
                                                           text_column: str,
                                                           ne_column: str,
                                                           target_df_export_path: Path,
                                                           replacement_strategies: List[str] = ["semantic_label_mask"]) -> Path:
        """
        Redacts private entity tokens in the specified text column of a DataFrame based on the given replacement strategies.
        
        :param ne_df: The DataFrame containing the text and named entities.
        :param text_column: The column name in the DataFrame that contains the text.
        :param ne_column: The column name in the DataFrame that contains the named entities as list of dictionaries.
        :param target_df_export_path: The path to export the redacted DataFrame.
        :param replacement_strategies: A list of strategies for replacing tokens.
               Options are - 
               (1) "semantic_label_mask" or (e.g. "maria martin" -> [PERSON])
               (2) "random_mask" or (e.g. "maria martin" [12] -> Xch6YTvb8mQz)
               (3) "generic_mask" or (e.g. "maria martin" -> XXXX)
        :return: The path to the exported redacted DataFrame.
        """
        redacted_ne_df = ne_df.copy()
        for strategy in replacement_strategies:
            redacted_column_name = f"{text_column}_redacted_with_{strategy}"
            redacted_ne_df[redacted_column_name] = None
            
            for idx, row in redacted_ne_df.iterrows():
                input_text = row[text_column]
                ne_list = json.loads(row[ne_column])
                private_entities, _ = TokenTreatmentUtils.filter_named_entities(ne_list)
                
                if private_entities:
                    redacted_text = TokenTreatmentUtils.redact_private_entity_tokens_in_text(
                        input_text=input_text,
                        private_entities=private_entities,
                        replacement_strategy=strategy
                    )
                    redacted_ne_df.at[idx, redacted_column_name] = redacted_text
        
        redacted_ne_df.to_parquet(target_df_export_path, index=False)
        return target_df_export_path
    
    @staticmethod
    def collect_private_entity_entities_for_dataframe(ne_df: pd.DataFrame,
                                                      ne_column: str,
                                                      target_column: str,
                                                      target_df_export_path: Path) -> Path:
        """
        Collects or filters private entities from the named entities column of a DataFrame
        
        :param ne_df: The DataFrame with named entities per rows.
        :param ne_column: The column name in the DataFrame that contains the named entities as list of dictionaries.
        :param target_column: The column name in the DataFrame to store the collected private entities as list of dictionaries.
        :param target_df_export_path: The path where the target dataframe with private entities will be saved.
        
        :return: The path to the exported DataFrame.
        """
        pe_df = ne_df.copy()

        total_private_entities = [0]
        def filter_and_count(data):
            ne_list = json.loads(data)
            pe_list, _ = TokenTreatmentUtils.filter_named_entities(ne_list)
            pe_count = len(pe_list) if isinstance(pe_list, list) else 0
            total_private_entities[0] += pe_count
            return json.dumps(pe_list)

        with tqdm(total=len(pe_df), desc=f"Collecting private entities") as pbar:
            pe_df[target_column] = pe_df[ne_column].apply(
                lambda data: (
                    pbar.update(1), 
                    filter_and_count(data), 
                    pbar.set_postfix({
                        "total_private_entities": total_private_entities[0]
                    })
                )[1]
            )
        
        pe_df = pe_df.drop(columns=[ne_column])
        pe_df.to_parquet(target_df_export_path, index=False)
        return target_df_export_path
    
    @staticmethod
    def update_private_entity_dataframe_with_stats(pe_df_file_path: Path, 
                                                   pe_column: str,
                                                   id_column: str) -> Dict[str, int]:
        """
        Collects statistics about private entities from the specified column of a DataFrame 
        and updates the DataFrame with these statistics.
        
        :param pe_df: The DataFrame containing private entities.
        :param pe_column: The column name in the DataFrame that contains the private entities as list of dictionaries.
        :param id_column: The column name in the DataFrame that contains the unique identifier for each row.
        
        :return: Returns a dictionary with counts of all private entity labels.
        """

        pe_df = pd.read_parquet(pe_df_file_path)
        
        id_label_counts: Dict[str, Dict[str, int]] = dict()
        all_labels: Set[str] = set()

        with tqdm(total=len(pe_df), desc=f"Collecting private entity stats") as pbar:
            for _, row in pe_df.iterrows():
                pe_list = json.loads(row[pe_column])
                row_label_counts: Dict[str, int] = dict()
                row_label_counts['pe_count_total'] = len(pe_list)
                for pe in pe_list:
                    label = pe['label']
                    if label not in row_label_counts:
                        row_label_counts[label] = 0
                    row_label_counts[label] += 1
                    all_labels.add(label)
                id_label_counts[row[id_column]] = row_label_counts
                pbar.update(1)
        
        all_label_counts: Dict[str, int] = dict()

        cols = ['pe_count_total'] + [f'pe_count_{label}' for label in all_labels]
        for col in cols:
            pe_df[col] = 0

        with tqdm(total=len(pe_df), desc=f"Updating dataframe with private entity stats") as pbar:
            for idx, row in pe_df.iterrows():
                itemid = row[id_column]
                row_label_counts = id_label_counts.get(itemid, {})
                total_count = row_label_counts.get('pe_count_total', 0)
                pe_df.at[idx, 'pe_count_total'] = total_count
                all_label_counts['pe_count_total'] = all_label_counts.get('pe_count_total', 0) + total_count
                for label in all_labels:
                    label_count = row_label_counts.get(label, 0)
                    pe_df.at[idx, f'pe_count_{label}'] = label_count
                    all_label_counts[label] = all_label_counts.get(label, 0) + label_count
                pbar.update(1)
        
        pe_df.to_parquet(pe_df_file_path, index=False)
        return all_label_counts