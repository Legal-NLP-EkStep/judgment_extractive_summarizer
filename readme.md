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
We empirically found out that "Arguments" , "Issues" and "Decision" written in original judgement are very crisp and rich in information. So we do not try to summarize them. We carry forward all the sentences with these 3 roles directly into the summary. "Preamble" is important in setting the context of case and also copied to summary.  For remaining rhetorical roles, we rank the sentences in descending order of importance as predicted by the AI model and choose the top ones as described in section 5. 


## 3. Data used for training summarizer model
Altohugh the style of writing the headnotes is not the best but it definitely captures important aspects of the judgment. We used the headnotes published along with supreme court judgements from 1950 to 1994. There are 7500 supreme court judgements which have headnotes.  We seperated the headnotes from judgment text. The headnotes are  abstractive summaries of judgement text. So we first found out the original judgment sentences from which headnotes were prepared. We used commonly used [heuristic](https://transformersum.readthedocs.io/en/latest/extractive/convert-to-extractive.html) of ROUGE maximization to convert these abstractive summaries to extractive summaries.  Each of these judgements was also passed through our [Rhetorical Roles Prediction model](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline#6-training-baseline-model-on-train-data) to get predicted rhetorical roles for each of the senteces. So finally in training data for each of the sentences, we have a flag indicating if this sentence is important to be captured in summary and its rhetorical role.
The intuition is that to decide if a sentences should be included in the summary, it is important not only to look at the words of the sentence but also the rhetorical role of that sentence. Certain rhetorical roles are more important than others and we hope that model would learn to capture this. 

## 4. AI model Architecture (BERTSUM RR)
We used [BERTSUMM](https://arxiv.org/pdf/1908.08345.pdf) architecture and modified it to include the sentence rhetorical role. We concatenated 768
dimensional sentence vector from CLS token to onehot encoded sentence rhetorical roles. The idea is thatif certain rhetorical roles are more important than others while creating summaries, then the model will learnthose. We call this model BERTSUM RR. The trained model is available here.


## 5. Using trained AI model to generate final summary
When we pass each of the sentence through RR model then we get predicted probability of each sentence to be in the summary. Since we want section wise summary, we take all the sentences that belong to a section & sort this list in decreasing order of predicted probability. Then want to pick certain sentences on top to create final summary for that section. The percentage of selected sentences depend on the lenght of the input judgment. Longer is the input judgement then lesser percentage of the sentences should be selected other wise summary becomes very long. Conversely, for shorted jugements the selection percentage should be higher to have enough sized summary. 


[Model file](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Extractive_summarization/model/model_headnotes/model.pt)

## 6. Results

## 7. Next Steps

