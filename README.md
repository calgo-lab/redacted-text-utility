# redacted-text-utility

This repository contains codebase for redacting sensitive information from 
text documents to check how different redaction process affects the utility 
of those documents when used in the downstream tasks.

For now, we are using - 

(1) Medical Intent Classification dataset from DATEXIS available at:
https://huggingface.co/datasets/DATEXIS/med_intent_classification

(2) As the texts are in English, an English NER model (based on 
xlm-roberta-large) fine-tuned on OntoNotes 5.0 from HuggingFace is used for 
redaction:
https://huggingface.co/flair/ner-english-ontonotes-large

Redacted datasets can be found at [here](data/processed/DATEXIS/med_intent_classification/):
<br>
(1) train-00000-of-00001.parquet > train-00000-of-00001_ne_redacted.parquet
<br>
(2) validation-00000-of-00001.parquet > validation-00000-of-00001_ne_redacted.parquet
<br>
(3) test-00000-of-00001.parquet > test-00000-of-00001_ne_redacted.parquet

Because NER models fine-tuned on OntoNotes 5.0 detects a lot of non-private 
entities we only redact entities of type: DATE, GPE, ORG and PERSON (GPE is 
short for Geo-Political Entity which includes locations).

Moreover, we also filter out some unusual DATE and PERSON entities.
Details of the implementation can be found [here](src/utils/token_treatment_utils.py).

For transperency, we keep a separate list of excluded date entities which can be found [here](data/processed/DATEXIS/med_intent_classification/).

In the redacted datasets, 3 new columns are added in regards to 3 different redaction strategies:
<br>
(1) "text_redacted_with_semantic_label_mask"
<br>
(2) "text_redacted_with_random_mask"
<br>
(3) "text_redacted_with_generic_mask"

Not all texts from all rows contain private entities. So, in case a text does not
contain any private entities, the row in those columns are kept empty.

Example:

File: train-00000-of-00001_ne_redacted.parquet
<br>
Row Index: 2106
<br>
[text]:
```
miss edwards is here for evaluation of facial pain this is a 54 -year-old male
```
[text_redacted_with_semantic_label_mask]:
```
miss [PERSON] is here for evaluation of facial pain this is a [DATE] male
```
[text_redacted_with_random_mask]:
```
miss lhyZXSX is here for evaluation of facial pain this is a vejE4fPRUxkG male
```
[text_redacted_with_generic_mask]:
```
miss XXXX is here for evaluation of facial pain this is a XXXX male
```

Following are the statistics of (T)otal found (P)rivate (E)ntities in the raw dataset:

| Data File                         |   T-Rows |   T-Rows-PE |   T-PE |   PERSON |   DATE |   GPE |   ORG |
|:----------------------------------|---------:|------------:|-------:|---------:|-------:|------:|------:|
| train-00000-of-00001.parquet      |     3886 |         396 |    642 |      460 |    151 |    16 |    15 |
| validation-00000-of-00001.parquet |      646 |          57 |     88 |       66 |     21 |     1 |     0 |
| test-00000-of-00001.parquet       |      760 |          72 |    117 |       93 |     23 |     0 |     1 |