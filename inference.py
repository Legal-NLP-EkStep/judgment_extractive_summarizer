import ast
import re
import os
from models.model_builder import ExtSummarizer
from models import data_loader
from prepro.data_builder import BertData, greedy_selection
from others.tokenization import BertTokenizer
from tqdm import tqdm
import torch
from spacy.lang.en import English
import math
import copy
import pandas as pd
import numpy as np

# from transformers import BertTokenizer

class ARGS():
    def __init__(self):
        self.default = True


class ExtractiveSummarizer:

    def __init__(self,model_path=None):
        super(ExtractiveSummarizer, self).__init__()
        self.initialized = False
        if model_path is None:
            print('Please provide model.pt path')
            exit()
        self.initialize(model_path)


    def _setargs_(self):
        parser = ARGS()
        parser.task = 'ext'
        parser.encoder = 'bert'
        parser.mode = 'test'
        parser.bert_data_path = 'src/'
        parser.model_path = 'src/'
        parser.result_path = 'src/'
        parser.temp_dir = 'src/'
        parser.batch_size = 5000
        parser.test_batch_size = 1
        parser.max_pos = 512
        parser.use_interval = True
        parser.large = False
        parser.load_from_extractive = ''
        parser.sep_optim = True
        parser.lr_bert = 2e-3
        parser.lr_dec = 2e-3
        parser.use_bert_emb = False
        parser.share_emb = False
        parser.finetune_bert = True
        parser.dec_dropout = 0.2
        parser.dec_layers = 6
        parser.dec_hidden_size = 768
        parser.dec_heads = 8
        parser.dec_ff_size = 2048
        parser.enc_hidden_size = 512
        parser.enc_ff_size = 512
        parser.enc_dropout = 0.2
        parser.enc_layers = 6
        # params for EXT
        parser.ext_dropout = 0.2
        parser.ext_layers = 2
        parser.ext_hidden_size = 768
        parser.ext_heads = 8
        parser.ext_ff_size = 2048
        parser.label_smoothing = 0.1
        parser.generator_shard_size = 32
        parser.alpha = 0.95
        parser.beam_size = 5
        parser.min_length = 0
        parser.max_length = 2000
        parser.max_tgt_len = 140
        parser.use_rhetorical_roles = True  #### whether to use rhetorical roles in the mode
        parser.seperate_summary_for_each_rr = False  #### whether to select top N sentences from each rhetorical role
        parser.rogue_exclude_roles_not_in_test = True  #### whether to remove the sections that are present in predicted summaries which are not in test data while ROGUE calculation
        parser.add_additional_mandatory_roles_to_summary = False  #### whether to add the additional mandatory roles to predicted summary
        parser.summary_sent_precent = 20  ##### top N percentage of sentences to be selected
        parser.use_adaptive_summary_sent_percent = True  ##### whether summary sentence percentage should be chosen as per input text sentence length
        parser.param_init = 0
        parser.param_init_glorot = True
        parser.optim = 'adam'
        parser.lr = 1
        parser.beta1 = 0.9
        parser.beta2 = 0.999
        parser.warmup_steps = 8000
        parser.warmup_steps_bert = 8000
        parser.warmup_steps_dec = 8000
        parser.max_grad_norm = 0
        parser.save_checkpoint_steps = 5
        parser.accum_count = 1
        parser.report_every = 1
        parser.train_steps = 1000
        parser.recall_eval = False
        parser.visible_gpus = '-1'
        parser.gpu_ranks = '0'
        parser.log_file = './log.log'
        parser.seed = 666
        parser.test_all = False
        parser.test_from = ''
        parser.test_start_from = -1
        parser.train_from = ''
        parser.report_rouge = True
        parser.block_trigram = True
        self.model_args = parser

        parser_preprocessing = ARGS()
        parser_preprocessing.pretrained_model = 'bert'

        parser_preprocessing.mode = 'format_to_bert'
        parser_preprocessing.select_mode = 'greedy'
        parser_preprocessing.map_path = 'src/'
        parser_preprocessing.raw_path = 'src/'
        parser_preprocessing.save_patt = './'

        parser_preprocessing.shard_size = 2000
        parser_preprocessing.min_src_nsents = 0
        parser_preprocessing.max_src_nsents = 50000
        parser_preprocessing.min_src_ntokens_per_sent = 0
        parser_preprocessing.max_src_ntokens_per_sent = 512
        parser_preprocessing.min_tgt_ntokens = 0
        parser_preprocessing.max_tgt_ntokens = 20000

        parser_preprocessing.lower = True
        parser_preprocessing.use_bert_basic_tokenizer = False

        parser_preprocessing.log_file = './log.log'

        parser_preprocessing.dataset = ''

        parser_preprocessing.n_cpus = 2
        self.preprocessing_args = parser_preprocessing

    def initialize(self, model_pt_path):
        """ Loads the models.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # try:
        #     self.db_obj = PostgresDatabase(host=None, database=None, user=None, password=None)
        # except:
        #     self.db_obj = None

        # Read model serialize/pt file
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the models.pt or pytorch_model.bin file")

        # Make necessary directories

        # Load model
        checkpoint = torch.load(model_pt_path, map_location=torch.device('cpu'))
        self._setargs_()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.model = ExtSummarizer(self.model_args, self.device, checkpoint)
        self.model.eval()

        self.initialized = True

    def check_token_authentication_and_update(self, token):
        if self.db_obj is not None:
            fetched_data = self.db_obj.fetch()
            count = [int(i[1]) for i in fetched_data if str(i[0]) == token]
            quota_used = [int(i[2]) for i in fetched_data if str(i[0]) == token]
            if not count:
                return False
            else:
                count = count[0]
                quota_used = quota_used[0]
                if quota_used < count:
                    quota_used = quota_used + 1
                    self.db_obj.update_request_count(token=token, request_count=count, quota_used=quota_used)
                    return True
                else:
                    return False
        else:
            return True

    def _preprocess_for_summarization(self, recieved_data):
        """
        recieved_data={"id":file_name,data{"text"},"annotations":[....]}
        :param recieved_data:
        :return bert_preprocessed_data:
        """
        import copy
        max_bert_tokens_per_chunk = 510

        source_chunk_id = 0
        doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [], 'source_filename': recieved_data['id'],
                    'sentence_id': [], 'src_chunk_id': source_chunk_id}
        doc_data_list = []

        for index, value in tqdm(enumerate(recieved_data['annotations'][0]['result']),
                                 total=len(recieved_data['annotations'][0]['result']), desc="preprocessing for bert"):
            value = value['value']
            if not value['labels'][0] in ['NONE', 'PREAMBLE']:
                tokenized = self.bert_tokenizer.tokenize(value['text'])
                sent_tokens = [token.text for token in self.tokenizer(value['text'])]
                if (sum([len(self.bert_tokenizer.tokenize(' '.join(i))) + 2 for i in doc_data['src']]) + len(
                        tokenized)) <= max_bert_tokens_per_chunk:
                    doc_data['src'].append(sent_tokens)
                    doc_data['src_rhetorical_roles'].append(value['labels'][0])
                    doc_data['sentence_id'].append(value['id'])
                elif len(
                        tokenized) > max_bert_tokens_per_chunk:
                    if doc_data['src']:
                        doc_data_list.append(copy.deepcopy(doc_data))
                        source_chunk_id += 1
                        doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                    'source_filename': recieved_data['id'],
                                    'sentence_id': [],
                                    'src_chunk_id': source_chunk_id}
                    tokens_list = [tokenized[i:i + max_bert_tokens_per_chunk] for i in
                                   range(0, len(tokenized), max_bert_tokens_per_chunk - 0)]
                    if len(tokens_list[-1]) < 100:
                        tokens_list = tokens_list[:-1]
                    misc_sentence_id = float(value['id'])
                    for _ in tokens_list:
                        misc_sentence_id += 0.01
                        sent_tokens = self.bert_tokenizer.convert_tokens_to_string(_).split(
                            ' ')  # [token.text for token in self.tokenizer(value['text'])]
                        doc_data['src'].append(sent_tokens)
                        doc_data['src_rhetorical_roles'].append(value['labels'][0])
                        doc_data['sentence_id'].append(misc_sentence_id)
                        doc_data_list.append(copy.deepcopy(doc_data))
                        source_chunk_id += 1
                        doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                    'source_filename': recieved_data['id'],
                                    'sentence_id': [],
                                    'src_chunk_id': source_chunk_id}
                else:
                    doc_data_list.append(copy.deepcopy(doc_data))
                    source_chunk_id += 1
                    doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                'source_filename': recieved_data['id'],
                                'sentence_id': [],
                                'src_chunk_id': source_chunk_id}
                    doc_data['src'].append(sent_tokens)
                    doc_data['src_rhetorical_roles'].append(value['labels'][0])
                    doc_data['sentence_id'].append(value['id'])

        if doc_data['src']:
            doc_data_list.append(copy.deepcopy(doc_data))

        return doc_data_list

    def _format_to_bert(self, bert_preprocessed_data):
        bert = BertData(self.preprocessing_args)
        datasets = []
        for d in tqdm(bert_preprocessed_data, desc="processing for input to model"):
            source, tgt = d['src'], d['tgt']
            f_name_chunk_id = '___'.join([d['source_filename'], str(d['src_chunk_id']), str(d['sentence_id'])])
            sent_labels = greedy_selection(source[:self.preprocessing_args.max_src_nsents], tgt, len(tgt))
            sent_rhetorical_roles = d['src_rhetorical_roles']
            if (self.preprocessing_args.lower):
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess(source, tgt, sent_labels, sent_rhetorical_roles,
                                     use_bert_basic_tokenizer=self.preprocessing_args.use_bert_basic_tokenizer,
                                     is_test=True)
            # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, sent_rr, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                           "src_sent_labels": sent_labels, "sentence_rhetorical_roles": sent_rr, "segs": segments_ids,
                           'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt, 'unique_id': f_name_chunk_id}
            datasets.append(b_data_dict)
        return datasets

    def _load_dataset(self, bert_tokenised_data):
        yield bert_tokenised_data

    def _preprocess(self, recieved_data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """

        # convert incoming data into label studio format
        self.preamble_text = recieved_data['data']['preamble_text']
        bert_formatted = self._preprocess_for_summarization(recieved_data)
        bert_formated_and_tokenized = self._format_to_bert(bert_formatted)
        data_iter = data_loader.Dataloader(self.model_args, self._load_dataset(bert_formated_and_tokenized),
                                           self.model_args.test_batch_size, self.device,
                                           shuffle=False, is_test=True)

        return data_iter

    def _inference(self, data_iter):
        """ Predict the class of a text using a trained transformer model.
        :param **kwargs:
        """

        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        self.model.eval()
        file_chunk_sent_scores = {}  ## key is filename and value is list of sentences containing sentence scores

        with torch.no_grad():
            for batch in tqdm(data_iter, desc="Running Inference"):
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sentence_rhetorical_roles = batch.sentence_rhetorical_roles
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, sentence_rhetorical_roles)
                sent_scores = sent_scores.cpu().data.numpy()
                file_name, chunk_id, sentence_ids = batch.unique_ids[0].split('___')
                chunk_id = int(chunk_id)
                sentence_ids = ast.literal_eval(sentence_ids)
                src_labels = list(labels.cpu().numpy()[0])
                if type(sent_scores[0]) == np.float32:
                    sent_scores = np.array([sent_scores])
                sent_scores_list = list(sent_scores[0])
                sent_rhetorical_roles_list = list(sentence_rhetorical_roles.cpu().data.numpy()[0])
                for sent_id, (sent_txt, sent_label, sent_score, sent_rhet_role,sentence_id) in enumerate(
                        zip(batch.src_str[0], src_labels, sent_scores_list, sent_rhetorical_roles_list,sentence_ids)):
                    if file_chunk_sent_scores.get(file_name) is None:
                        file_chunk_sent_scores[file_name] = []
                    sent_dict = {'file_name': file_name, 'chunk_id': chunk_id, 'sent_id': sentence_id, 'sent_txt': sent_txt,
                                 'sent_score': sent_score, 'sent_label': sent_label,
                                 'sent_rhetorical_role': sent_rhet_role}
                    file_chunk_sent_scores[file_name].append(sent_dict)

        return file_chunk_sent_scores

    def _postprocess(self, inference_output):
        ##### Original mapping of rhetorical roles to ids is done in  data_builder.py
        # self.rhetorical_role_map = {'FAC':1,'RLC':2,'ARG_PETITIONER':3,'ARG_RESPONDENT':4,'ISSUE':5,'PRE_RELIED':6,'PRE_NOT_RELIED':7,
        #                                 'ANALYSIS':8,'STA':9,'RATIO':10,"RPC":11}
        # Now we are mapping these to LawBriefs Roles
        lawbriefs_summary_map = {1: "facts", 2: "facts", 3: "arguments", 4: "arguments",
                                 5: "issue", 6: "ANALYSIS", 7: 'ANALYSIS',
                                 8: 'ANALYSIS', 9: 'ANALYSIS', 10: 'ANALYSIS',
                                 11: 'decision'}  ##### keys are baseline rhetorical roles and values are LawBriefs roles.
        predicted_categories = ['facts', 'issue', 'arguments', 'ANALYSIS', 'decision']
        additional_mandaroty_categories = ['issue', 'decision']

        def _process_sentences_with_scores_and_add_percentile_ranks(output_inference, text_summary):
            temp = copy.deepcopy(output_inference)
            f_name = list(temp.keys())[0]
            sent_scores_list = temp[f_name]
            for sent_dict in sent_scores_list:
                sent_dict['sent_rhetorical_role'] = lawbriefs_summary_map[sent_dict['sent_rhetorical_role']].upper()
                if sent_dict['sent_txt'] in text_summary:
                    sent_dict['sent_label'] = 1
                # sent_dict['sent_id'] = str(sent_dict['chunk_id']) + "_" + str(sent_dict['sent_id'])
                # del sent_dict['chunk_id']

            df = pd.DataFrame(sent_scores_list)
            df['Percentile_Rank'] = df.sent_score.rank(pct=True)
            return df.to_dict('records')

        def get_adaptive_summary_sent_percent(sent_cnt):
            ######## get the summary sentence percentage to keep in output based in input sentence cnt. The values are found by piecewise linear regression
            if sent_cnt <= 77:
                const = 40.5421
                slope = -0.2444
            elif sent_cnt <= 122:
                const = 29.5264
                slope = -0.1013
            else:
                const = 17.8994
                slope = - 0.006
            summary_sent_precent = slope * sent_cnt + const
            return summary_sent_precent

        def create_concatenated_summaries(file_chunk_sent_scores, use_adaptive_summary_sent_percent=True,
                                          summary_sent_precent=30, use_rhetorical_roles=True,
                                          seperate_summary_for_each_rr=True,
                                          add_additional_mandatory_roles_to_summary=False):
            predicted_rr_summaries = []
            #### this function accepts the sentence scores and returns predicted summary
            for file_name, sent_list_all in file_chunk_sent_scores.items():
                ####### remove sentences without single alphabet
                sent_list = [sent for sent in sent_list_all if re.search('[a-zA-Z]', sent['sent_txt'])]
                if use_adaptive_summary_sent_percent:
                    summary_sent_precent = get_adaptive_summary_sent_percent(len(sent_list))
                else:
                    summary_sent_precent = summary_sent_precent

                if use_rhetorical_roles and seperate_summary_for_each_rr:
                    # ######## take top N sentences for each rhetorical role
                    file_rr_sents = {}  ##### keys are rhetorical roles and values are dict of {'sentences':[],'token_cnt':100}
                    for sent_dict in sent_list:
                        sent_token_cnt = len(sent_dict['sent_txt'].split())
                        sent_rr = lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
                        if file_rr_sents.get(sent_rr) is None:
                            file_rr_sents[sent_rr] = {'sentences': [sent_dict], 'token_cnt': sent_token_cnt}
                        else:
                            file_rr_sents[sent_rr]['sentences'].append(sent_dict)
                            file_rr_sents[sent_rr]['token_cnt'] += sent_token_cnt

                    min_token_cnt_per_rr = 50  ######## if original text for a rhetorical role is below this then it is not summarized.
                    selected_sent_list = []
                    for rr, sentences_dict in file_rr_sents.items():
                        if sentences_dict['token_cnt'] <= min_token_cnt_per_rr or rr in additional_mandaroty_categories:
                            selected_sent_list.extend(sentences_dict['sentences'])
                        else:
                            rr_sorted_sent_list = sorted(sentences_dict['sentences'], key=lambda x: x['sent_score'],
                                                         reverse=True)

                            sents_to_keep = math.ceil(summary_sent_precent * len(sentences_dict['sentences']) / 100)
                            rr_selected_sent = rr_sorted_sent_list[:sents_to_keep]
                            rr_selected_sent = sorted(rr_selected_sent, key=lambda x: (x['chunk_id'], x['sent_id']))
                            selected_sent_list.extend(rr_selected_sent)

                else:
                    ######### take top N sentences by combining all the rhetorical roles
                    sent_list = sorted(sent_list, key=lambda x: x['sent_score'], reverse=True)
                    sents_to_keep = math.ceil(summary_sent_precent * len(sent_list) / 100)
                    selected_sent_list = sent_list[:sents_to_keep]
                    selected_sent_list = sorted(selected_sent_list, key=lambda x: (x['chunk_id'], x['sent_id']))

                predicted_summary_rr = {}  ## keys are rhetorical role and values are concatenated sentences
                ## create predicted summary
                for sent_dict in selected_sent_list:
                    sent_lawbriefs_role = lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
                    if predicted_summary_rr.get(sent_lawbriefs_role) is None:
                        predicted_summary_rr[sent_lawbriefs_role] = sent_dict['sent_txt']
                    else:
                        predicted_summary_rr[sent_lawbriefs_role] = predicted_summary_rr[sent_lawbriefs_role] + '\n' + \
                                                                    sent_dict['sent_txt']

                ######## copy the additional mandatory roles to summary
                if use_rhetorical_roles and add_additional_mandatory_roles_to_summary and not seperate_summary_for_each_rr:
                    sent_list = sorted(sent_list, key=lambda x: (x['chunk_id'], x['sent_id']))
                    for category in additional_mandaroty_categories:
                        category_sentences = [i for i in sent_list if
                                              lawbriefs_summary_map[i['sent_rhetorical_role']] == category]
                        if category_sentences:
                            if predicted_summary_rr.get(category) is not None:
                                ###### remove the category as it may not have all the sentences.
                                predicted_summary_rr.pop(category)
                            for cat_sent in category_sentences:
                                if predicted_summary_rr.get(category) is None:
                                    predicted_summary_rr[category] = cat_sent['sent_txt']
                                else:
                                    predicted_summary_rr[category] = predicted_summary_rr[category] + '\n' + \
                                                                     cat_sent['sent_txt']
                predicted_rr_summaries.append(predicted_summary_rr)

            return predicted_rr_summaries

        rr_summaries = create_concatenated_summaries(inference_output)
        rr_summaries[0]['PREAMBLE'] = self.preamble_text
        summary_text = ' '.join([rr_summaries[0][key] for key in rr_summaries[0].keys()])
        rr_summaries[0]['all_sentences_with_scores'] = _process_sentences_with_scores_and_add_percentile_ranks(
            inference_output, summary_text)
        return rr_summaries

    def infer(self,rhetorical_role_data):
        iter = self._preprocess(rhetorical_role_data)
        out = self._inference(iter)
        processed_output = self._postprocess(out)
        return processed_output