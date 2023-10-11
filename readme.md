### Please use this repository if you are interested in training your own extractive summarizer model and understanding details of [Corpus for Automatic Structuring of Legal Documents](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.470.pdf). If you want to use pre-trained models on your own data then please use [opennyai library](https://github.com/OpenNyAI/Opennyai).

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
| Analysis  | Analysis, Statute, Precedent Relied, Precedent Not Relied, Ratio of the decision   |
| Decision  | Ruling by Present Court  |

We believe this structure of writing summaries is better suited for Legal Research and Infomation Extraction. This will also improve the readability of the summaries.
### 2.1 Extractive summarization using Rhetorical Roles
There are two styles of creating summaries viz. Extractive & Abstractive. Extractive summaries pick up important sentences as-is and put them in order for creating final summary. Abstractive summarization on the other hand paraphrases the important information to create crisp summary in its own words. While abstractive summaries are more useful, they are harder to create and evaluate. Hence, as first step we focus on extractive summarization which will pick up most important sentences and arrange them in the structure described above. Once this task is done correctly, then we can focus on the abstractive summarization
### 2.2 Which rhetorical roles are summarized?
We empirically found out that "Issues" and "Decision" written in original judgement are very crisp and rich in information. So we do not try to summarize them. We carry forward all the sentences with these 2 roles directly into the summary. "Preamble" is important in setting the context of case and also copied to summary.  For remaining rhetorical roles, we rank the sentences in descending order of importance as predicted by the AI model and choose the top ones as described in section 5. 


## 3. [Data](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/judgement_extractive_summarizer/data/data.zip) used for training summarizer [model](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/model/model_headnotes/model.pt)
Although the style of writing the headnotes is not the best but it definitely captures important aspects of the judgment. We used the headnotes published along with supreme court judgements from 1950 to 1994. After filtering for minimum lengths there are 10440 supreme court judgements which have headnotes. We split them randomly into 9337 train & 503 test judgments.  We seperated the headnotes from judgment text. The headnotes are  abstractive summaries of judgement text. So we first found out the original judgment sentences from which headnotes were prepared. We used commonly used [heuristic](https://transformersum.readthedocs.io/en/latest/extractive/convert-to-extractive.html) of ROUGE maximization to convert these abstractive summaries to extractive summaries.  Each of these judgements was also passed through our [Rhetorical Roles Prediction model](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline#6-training-baseline-model-on-train-data) to get predicted rhetorical roles for each of the senteces. So finally in training data for each of the sentences, we have a flag indicating if this sentence is important to be captured in summary and its rhetorical role.
The intuition is that to decide if a sentences should be included in the summary, it is important not only to look at the words of the sentence but also the rhetorical role of that sentence. Certain rhetorical roles are more important than others and we hope that model would learn to capture this. 

## 4. AI model Architecture (BERTSUM RR)
We used [BERTSUMM](https://arxiv.org/pdf/1908.08345.pdf) architecture and modified it to include the sentence rhetorical role. We concatenated 768
dimensional sentence vector from CLS token to onehot encoded sentence rhetorical roles. The idea is thatif certain rhetorical roles are more important than others while creating summaries, then the model will learnthose. We call this model BERTSUM RR. The trained model is available here.


## 5. Using trained AI model to generate final summary
When we pass each of the sentence through RR model then we get predicted probability of each sentence to be in the summary. Since we want section wise summary, we take all the sentences that belong to a section & sort this list in decreasing order of predicted probability. Then want to pick certain sentences on top to create final summary for that section. The percentage of selected sentences depend on the lenght of the input judgment. Longer is the input judgement then lesser percentage of the sentences should be selected other wise summary becomes very long. Conversely, for shorted jugements the selection percentage should be higher to have enough sized summary. The following graph shows scatterplot of number of sentences in input text on x axis and percentage of selected senteces in summary on y axis.

![image](https://user-images.githubusercontent.com/4078857/176412454-3fc44d4f-7c12-42fa-8282-99d4e4d46cf2.png)

We fitted a piecewise linear regression(orange line) to predict the percentage of summary sentences based on the input sentences count.

## 6. Results 
We tested the BERTSUM RR model (trained on 9337 judgements) on the 503 test judgments. The ROUGE scores are as below.

| ROUGE 1  | ROUGE2 | ROUGE L |
| ------------- | ------------- | ------------- |
|  0.6328 |	0.4152 |	0.6219 | 

## 7. Creating summaries of custom judgments
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

## 8. Running summarizer on sample
This sample file is generated after passing it through our rhetorical role [repo](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline). To run the extractive summarizer on any judgement follow the given steps:


1. Pass judgement through rhetorical role repo
2. Use the given code to run extractive [model](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/model/model_headnotes/model.pt)

```python
from inference import ExtractiveSummarizer
import json
summ = ExtractiveSummarizer('./model.pt')
rr_data = json.load(open('./sample_rhetorical_output.json'))
summary = summ.infer(rr_data[0])
```

## 9. Conclusion & Next Steps
We have trained an extractive summarizer model for Indian Court judgments. We believe that this model will keep on improving with human in the loop feedback. This model in current form can also give decent summaries as shown by the results but needs caution while application.

## Acknowledgements
This work is part of [OpenNyAI](https://opennyai.org/) mission which is funded by [EkStep](https://ekstep.org/) and [Agami](https://agami.in/). 

We thank Aditya Gor from lawbriefs for his valuable feeback about summaries and sharing the data for summaries created by the students which inspired the current summary structure. You can get the raw data by requesting on this [email](adityagor282@gmail.com).

We also thank the volunteers from Manupatra and OpenNyAI for evaluating summaries and sharing valuable feedback.
