# Artifical Intelligence augmented Summarization of Indian Court Judgments

## 1. Why use Artificial Intelligence to help humans create faster summaries of court judgments?
Court judgments can be very long and it is a common practice for legal publishers to create headnotes of judgments. E.g. [sample headnote](https://main.sci.gov.in/judgment/judis/5268.pdf).
The process of creating headnotes is  manual and based on the certain rules and patterns. With advances in Artifical Intelligence, we can create automatically summaries of long text and then an expert to correct it to create final summary. This will drastically reduce the time needed for creation of headnotes and make the process more consistent. AI model can also learn from the feedback given by the expert and keep on improving the results.

## 2. Structure of Judgment Summary
While standard way of writing headnotes captures the important aspects of the judgement like HELD, experts believe that it is not the best style of writing summaries. E.g. it is difficult to establish if the facts of a new case are similar to facts of an old case by reading headnotes of the old case.

So we have come up with revised structure of writing summaries. Summary will have 5 sections Facts summary, Arguments summary, Issue summary, Analysis Summary and Decision Summary. Leveraging our previous work on [structuring court judgements](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline), we can automatically predict Rhetorical Roles for each sentence and then create this sectionwise summary. The following table shows which rhetorical roles to expect in each of the summary sections
| Summary Section  | Rhetorical Roles |
| ------------- | ------------- |
| Facts  | Facts, Ruling by Lower Court  |
| Issue  | Issues  |
| Arguments  | Argument by Petitioner, Argument by Respondent  |
| Analysis  | Analysis,Statute,Precedent Relied,Precedent Not Relied,Ratio of the decision   |
| Decision  | Ruling by Present Court  |

Extractive Text Summarization is the task of extracting important information or sentence from the given text.
Extractive summarizers are so-called because they select sentences from the originally given text passage to create the
summary. 

## 2. Extractive summarization using Rhetorical Roles

We wanted to see how Rhetorical Roles could help generate better summaries. Towards this goal, we passed the sentences
through bertsumm where sentences were selected to create the summaries. To see the impact of rhetorical roles we
provided sentences with the predicted rhetorical roles from our model. This helped in generating summaries of various
roles separately. This approach improved the generated summaries.

## 3. Extractive summarization trained model file

[Model file](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/model/model_headnotes/model.pt)

## 4. Extractive summarization data


# Setup

**Python version**: This code is in Python3.7

```pip install -r requirements.txt```

## Data Preparation

### Option 1: Download the processed data. Data has already been processed in .pt files that you can use directly.

[Pre-processed data](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/data/data.zip)

unzip the zipfile

### Option 2: process the data yourself

#### Step 1 Download Stories

Download and unzip the `sample json` file
from [here](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/data/sample.json)
. This a small set of our train data for reference.

#### Note: Replace with your data

#### Step 2. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -lower -n_cpus 1 -log_file ../logs/preprocess.log max_src_nsents 500000 -min_src_nsents 1 -min_src_ntokens_per_sent 0 -max_src_ntokens_per_sent 512 -min_tgt_ntokens 0 -max_tgt_ntokens 200000 
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python3 train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -report_every 50 -save_checkpoint_steps 500 -batch_size 5000 -train_steps 5000 -accum_count 2 -log_file ../logs/ext_bert_3jan -use_interval true -warmup_steps 1000 -max_pos 512 -use_rhetorical_roles true
```

## Model Evaluation
```
 python3 train.py -task ext -mode test -test_from /data/bertsum/model.pt -batch_size 5000 -test_batch_size 1 -bert_data_path BERT_DATA_PATH -log_file ../logs/bertsum -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 2000 -alpha 0.95 -min_length 0 -result_path ../logs/ -use_rhetorical_roles true -rogue_exclude_roles_not_in_test true -add_additional_mandatory_roles_to_summary true  -use_adaptive_summary_sent_percent true 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)

