from typing import Optional

from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings.transformer import (
    truncate_hidden_states,
    combine_strided_tensors,
    document_cls_pooling,
    document_mean_pooling,
    document_max_pooling,
    fill_masked_elements,
    fill_mean_token_embeddings
)

import flair
import torch


class LongformerDocumentEmbeddings(TransformerDocumentEmbeddings):

    def forward(
        self,
        input_ids: torch.Tensor,
        sub_token_lengths: Optional[torch.LongTensor] = None,
        token_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        overflow_to_sample_mapping: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ):
        model_kwargs = {}
        if langs is not None:
            model_kwargs["langs"] = langs
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        if bbox is not None:
            model_kwargs["bbox"] = bbox
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values

        ### Longformer-specific fix
        if self.model.config.model_type == "longformer":
            # ensure input length does not exceed Longformer max
            max_len = self.model.config.max_position_embeddings
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, :max_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :max_len]
                if word_ids is not None:
                    word_ids = word_ids[:, :max_len]

            # create global attention mask with dtype long and correct device
            global_attention_mask = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
            global_attention_mask[:, 0] = 1  # CLS token
            model_kwargs["global_attention_mask"] = global_attention_mask
        
        hidden_states = self.model(input_ids, **model_kwargs)[-1]
        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)

        # for multimodal models like layoutlmv3, we truncate the image embeddings as they are only used via attention
        hidden_states = truncate_hidden_states(hidden_states, input_ids)

        # only use layers that will be outputted
        hidden_states = hidden_states[self.layer_indexes, :, :]
        if self.layer_mean:
            hidden_states = hidden_states.mean(dim=0)
        else:
            hidden_states = torch.flatten(hidden_states.permute((0, 3, 1, 2)), 0, 1).permute((1, 2, 0))

        if self._can_document_embedding_shortcut():
            return {"document_embeddings": hidden_states[:, 0]}

        if self.allow_long_sentences:
            assert overflow_to_sample_mapping is not None
            sentence_hidden_states = combine_strided_tensors(
                hidden_states, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, 0
            )
            if self.tokenizer.is_fast and self.token_embedding:
                word_ids = combine_strided_tensors(
                    word_ids, overflow_to_sample_mapping, self.stride // 2, self.tokenizer.model_max_length, -100
                )
        else:
            sentence_hidden_states = hidden_states

        result = {}

        if self.document_embedding:
            if self.cls_pooling == "cls" and self.initial_cls_token:
                document_embeddings = sentence_hidden_states[:, 0]
            else:
                assert sub_token_lengths is not None
                if self.cls_pooling == "cls":
                    document_embeddings = document_cls_pooling(sentence_hidden_states, sub_token_lengths)
                elif self.cls_pooling == "mean":
                    document_embeddings = document_mean_pooling(sentence_hidden_states, sub_token_lengths)
                elif self.cls_pooling == "max":
                    document_embeddings = document_max_pooling(sentence_hidden_states, sub_token_lengths)
                else:
                    raise ValueError(f"cls pooling method: `{self.cls_pooling}` is not implemented")
            result["document_embeddings"] = document_embeddings

        if self.token_embedding:
            assert word_ids is not None
            assert token_lengths is not None
            all_token_embeddings = torch.zeros(  # type: ignore[call-overload]
                word_ids.shape[0],
                token_lengths.max(),
                self.embedding_length_internal,
                device=flair.device,
                dtype=sentence_hidden_states.dtype,
            )
            true_tensor = torch.ones_like(word_ids[:, :1], dtype=torch.bool)
            if self.subtoken_pooling == "first":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                first_mask = torch.cat([true_tensor, gain_mask], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, first_mask, word_ids, token_lengths
                )
            elif self.subtoken_pooling == "last":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                last_mask = torch.cat([gain_mask, true_tensor], dim=1)
                all_token_embeddings = fill_masked_elements(
                    all_token_embeddings, sentence_hidden_states, last_mask, word_ids, token_lengths
                )
            elif self.subtoken_pooling == "first_last":
                gain_mask = word_ids[:, 1:] != word_ids[:, : word_ids.shape[1] - 1]
                first_mask = torch.cat([true_tensor, gain_mask], dim=1)
                last_mask = torch.cat([gain_mask, true_tensor], dim=1)
                all_token_embeddings[:, :, : sentence_hidden_states.shape[2]] = fill_masked_elements(
                    all_token_embeddings[:, :, : sentence_hidden_states.shape[2]],
                    sentence_hidden_states,
                    first_mask,
                    word_ids,
                    token_lengths,
                )
                all_token_embeddings[:, :, sentence_hidden_states.shape[2] :] = fill_masked_elements(
                    all_token_embeddings[:, :, sentence_hidden_states.shape[2] :],
                    sentence_hidden_states,
                    last_mask,
                    word_ids,
                    token_lengths,
                )
            elif self.subtoken_pooling == "mean":
                all_token_embeddings = fill_mean_token_embeddings(
                    all_token_embeddings, sentence_hidden_states, word_ids, token_lengths
                )
            else:
                raise ValueError(f"subtoken pooling method: `{self.subtoken_pooling}` is not implemented")

            result["token_embeddings"] = all_token_embeddings
        return result
