from typing import Any, Dict, List, Tuple

from flair.data import Sentence

from core.logging import get_logger
from infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from infrastructure.frameworks.sentence_with_boundary import SentenceWithBoundary
from infrastructure.frameworks.sequence_tagger_loader import SequenceTaggerLoader
from infrastructure.frameworks.somajo_tokenizer import SoMaJoTokenizer

import json
import re


class SequenceTaggerInferenceMaker(ModelInferenceMaker):
    """
    This class implements the infer method to return the inference result with 
    a Flair SequenceTagger model.
    """
    
    _somajo_tokenizer: SoMaJoTokenizer = None

    def __init__(self, model_loader: SequenceTaggerLoader):
        """
        Initialize the SequenceTaggerInferenceMaker with a model loader.
        
        :param model_loader: The SequenceTaggerLoader instance to load the model.
        """
        super().__init__(model_loader)
        self.logger = get_logger(__name__)
        self._load_somajo_tokenizer()
    
    def _load_somajo_tokenizer(self) -> None:
        """
        Load the SoMaJo tokenizer.

        :return: None
        """
        if self._somajo_tokenizer is None:
            self._somajo_tokenizer = SoMaJoTokenizer()

    def infer(self, input_text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Make inference using the loaded SequenceTagger model.
        
        :param input_text: The input text for which inference is to be made.
        :param **kwargs: Additional keyword arguments for inference.
        :return: The inference result as a list of entity dict objects.
        """
        tagger = self.model_loader.load()
        sentences: List[SentenceWithBoundary] = self._get_sentences_with_boundaries(input_text)
        flair_sentences = [swb.sentence for swb in sentences]
        tagger.predict(flair_sentences)
        return self._convert_flair_sentences_to_list_of_entity_dict(input_text, sentences)

    def _get_sentences_with_boundaries(self, original_text: str, buffer: int = 15) -> List[SentenceWithBoundary]:
        """
        Get flair sentences built with SoMaJo tokens along with sentence boundaries in the original text.
        
        :param original_text: The original text to process.
        :param buffer: The buffer size to use for sentence boundary detection.
        :return: List of SentenceWithBoundary objects
        """
        sentences_with_boundaries: List[SentenceWithBoundary] = list()

        ### Replace tab characters with a unique placeholder to prevent tokenization issues.
        
        ### Example scenario:
        
        ### Janina Parkinson MD Msc		Dr.med. Veronica Kugic		Prof. Dr.Dr. Konstantin Lauterbach
        ### Stationsärztin			    Oberärztin			        Chefarzt
        
        ### This is a typical ending of a clinical note where tab characters are used for alignment.
        ### Any whitespace character (including tab) is treated same in SoMaJo tokenizer
        ### In this particular case, MD Msc Dr. would be tokenized as a single token which is undesirable.

        text_for_tokenization = original_text.replace('\t', '!!!')
        sentences = self._somajo_tokenizer.tokenizer.tokenize_text([text_for_tokenization])
        offset: int = 0
        prev_end: int = 0
        for _, sentence in enumerate(sentences):
            potential_sentence_of_original_form = ""
            for token in sentence:
                token_text = token.text.replace('!!!', '')
                potential_sentence_of_original_form += token_text + (' ' if token.space_after and not token.last_in_sentence else '')
            
            start, end = self._get_potential_sentence_boundary(original_text, potential_sentence_of_original_form, offset, buffer)
            if start - prev_end > buffer:
                start, end = self._get_potential_sentence_boundary(original_text, potential_sentence_of_original_form, prev_end - buffer, buffer)
            prev_end = end
            
            sentences_with_boundaries.append(SentenceWithBoundary(Sentence(potential_sentence_of_original_form), start, end))
            offset = end
        
        if len(sentences_with_boundaries) > 1:
            idx = 0
            while idx < len(sentences_with_boundaries):
                swb = sentences_with_boundaries[idx]
                if swb.start == swb.end:
                    start_idx = idx
                    end_idx = idx
                    while end_idx + 1 < len(sentences_with_boundaries) and \
                        sentences_with_boundaries[end_idx + 1].start == sentences_with_boundaries[end_idx + 1].end:
                        end_idx += 1
                    
                    if start_idx > 0:
                        new_start = sentences_with_boundaries[start_idx - 1].end
                        for fix_idx in range(start_idx, end_idx + 1):
                            sentences_with_boundaries[fix_idx].start = max(new_start - buffer, 0)
                    
                    if end_idx + 1 < len(sentences_with_boundaries):
                        new_end = sentences_with_boundaries[end_idx + 1].start
                        for fix_idx in range(start_idx, end_idx + 1):
                            sentences_with_boundaries[fix_idx].end = min(new_end -1, len(original_text))
                    idx = end_idx + 1
                else:
                    idx += 1
        
        return sentences_with_boundaries
    
    def _get_potential_sentence_boundary(self,
                                         original_text: str,
                                         somajo_tokenized_sentence: str,
                                         offset: int = 0,
                                         buffer: int = 15) -> Tuple[int, int]:
        """
        Find sentence boundaries in the original text based on the SoMaJo tokenized sentence.
        
        :param original_text: The original text.
        :param somajo_tokenized_sentence: The SoMaJo tokenized sentence as a single string.
        :param offset: The offset to start searching from.
        :param buffer: Additional buffer length to consider when searching for the end token.
        :return: A tuple (start_index, end_index) representing the sentence boundaries.
        """
        tokens = [token for token in somajo_tokenized_sentence.split(' ') if token]
        if not tokens:
            return offset, offset

        start_stretch = tokens[0]
        final_stretch = tokens[-1]

        start_index = original_text.find(start_stretch, offset)
        if start_index == -1:
            return offset, offset
        
        expected_length = len(somajo_tokenized_sentence)
        search_end = min((start_index + expected_length + buffer), len(original_text))
    
        final_index = -1
        temp_offset = start_index
        
        while temp_offset < search_end:
            pos = original_text.find(final_stretch, temp_offset)
            if pos == -1 or pos + len(final_stretch) > search_end:
                break

            if pos >= start_index:
                temp_offset = pos + len(final_stretch)
                final_index = pos
            else:
                temp_offset = pos + 1

        if final_index != -1:
            final_index += len(final_stretch)
        else:
            final_index = min(start_index + expected_length, len(original_text))
            
        return start_index, final_index

    def _convert_flair_sentences_to_list_of_entity_dict(self, 
                                                        input_text: str, 
                                                        model_output: List[SentenceWithBoundary]) -> List[Dict[str, Any]]:
        """
        Convert SentenceWithBoundary objects to aggregated list of entity dict objects.
        
        :param input_text: The original input text.
        :param model_output: The model output containing SentenceWithBoundary objects.
        :return: A list of entity dict objects.
        """
        items: List[Dict[str, Any]] = list()
        token_id = 0
        for swb in model_output:
            next_cursor = swb.start
            sentence = swb.sentence
            self.logger.info(f'{sentence.to_tagged_string()}')
            labels = sentence.get_labels()
            for label in labels:
                text = label.data_point.text
                start = input_text.find(text, next_cursor, swb.end)

                if start == -1 and re.search(r'\s', text):
                    
                    tokens = text.split()
                    if len(tokens) >= 2:
                        first_token = tokens[0]
                        last_token = tokens[-1]
                        first_pos = input_text.find(first_token, next_cursor, swb.end)
                        
                        if first_pos != -1:
                            last_pos = input_text.find(last_token, first_pos + len(first_token), swb.end)
                            
                            if last_pos != -1:
                                boundary_start = first_pos
                                boundary_end = last_pos + len(last_token)
                                extracted_text = input_text[boundary_start: boundary_end]
                                
                                normalized_extracted = re.sub(r'\s+', ' ', extracted_text)
                                normalized_predicted = re.sub(r'\s+', ' ', text)
                                
                                if normalized_extracted == normalized_predicted:
                                    start = boundary_start
                                    end = boundary_end
                                    token_id += 1
                                    next_cursor = end
                                    
                                    items.append({
                                        'token_id': 'T'+str(token_id), 
                                        'label': label.value, 
                                        'start': start, 
                                        'end': end, 
                                        'token': input_text[start: end]
                                    })
                elif start != -1:
                    end = start + len(text)
                    token_id += 1
                    next_cursor = end
                    
                    items.append({
                        'token_id': 'T'+str(token_id), 
                        'label': label.value, 
                        'start': start, 
                        'end': end, 
                        'token': input_text[start: end]
                    })
                else:
                    token_id += 1
                    items.append({
                        'token_id': 'T'+str(token_id), 
                        'label': label.value, 
                        'start': -1, 
                        'end': -1, 
                        'token': text
                    })

        self.logger.info(json.dumps(items, indent=2, ensure_ascii=False))
        return items