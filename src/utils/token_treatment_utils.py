from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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