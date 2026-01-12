# redacted-text-utility

This repository contains codebase for redacting sensitive information from 
text documents to check how different redaction process affects the utility 
of those documents when used in the downstream tasks.

## Medical Intent Classification Dataset ([DATEXIS](https://huggingface.co/DATEXIS))

Available at:
https://huggingface.co/datasets/DATEXIS/med_intent_classification

### Preview
| text  | intents |
|-------|---------|
| you do have a little bit of periphe- peripheral neuropathy . um , there is a medication we can use if they get really bad , but you're already on so many medications . | ["Discussion", "Medication", "Reassessment"] |
| and where would you say the tingling and numbness is ? | ["Acute Symptoms"] |
| doctor: alright thanks good seeing you thanks for coming in to them | ["Chitchat"] |

### Downstream Task
Medical Intent Classification is a multi-label classification task where 
given a medical text, the goal is to predict one or more medical 
intents/labels associated with that text.

### Redaction Model
As the texts are in English, an English NER model (based on 
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

### Results
\* Experiments done, results will be compiled and documented soon.

## European Court of Human Rights Dataset ([AUEB-NLP](https://huggingface.co/AUEB-NLP))

Available at: https://huggingface.co/datasets/glnmario/ECHR

The dataset is an adoptation of the original ECHR dataset introduced by 
Chalkidis et al. (2019): [Neural Legal Judgment Prediction in English](https://aclanthology.org/P19-1424/)

*The original dataset download [link](https://archive.org/details/ECHR-ACL2019) 
from the paper or the [link](http://archive.org/details/ECtHR-NAACL2021/) from 
[HuggingFace](https://huggingface.co/datasets/AUEB-NLP/ecthr_cases) does not 
work anymore and hence this adoptation is used in our experiments. (last checked 
on: 9th January 2026)*

The dataset contains approximately 11.5k cases from ECHR’s public database. 
For each case, the dataset provides a list of facts (column - "text") and 
a binary label (column - "binary_judgement") indicating whether any human 
rights article or protocol of European Convention of Human Rights has been 
violated (1) or not (0).

### Preview
| itemid | text  | binary_judgement |
|--------|-------|------------------|
| 001-4817 | The applicant is a British national, born in 1945 and living in Rome. The facts of the case, as submitted by the parties, may be summarised as follows. The applicant's ... | 0 |
| 001-89307 | 7. The applicant, Mrs Danutė Balsytė-Lideikienė, is a Lithuanian national, who was born in 1947. At present she lives in Lithuania. 8. The applicant is the founder and ... | 1 |

### Downstream Task - 1: Binary Violation Prediction
Binary Violation Prediction is a binary classification task where given the 
facts of a case, the goal is to predict whether any human rights article or 
protocol of European Convention of Human Rights has been violated (1) or 
not (0).

### Preprocessing and Sample Selection
The adoptated dataset contains some cases with very large texts (more than
5.5k tokens). Such cases are excluded from the experiments to avoid
memory issues during model training. So the samples with tokens count between 
512 and 10x512 are selected for the experiments that ensures every text 
contains a few private entities to redact while also avoiding memory issues.

The following table and histogram shows the distribution of number of tokens in the text column for the ECHR dataset without any sampling -

| doc_num | mean | std  | min | 25% | 50%  | 75%  | 90%   | max    |
|--------:|-----:|-----:|----:|----:|-----:|-----:|------:|-------:|
|   11478 | 2538 | 2924 |  14 | 818 | 1737 | 3184 |  5511 |  59784 |

![ECHR_Dataset_num_tokens_distribution](plots/glnmario/ECHR/eda/ECHR_Dataset_num_tokens_distribution.jpg)

