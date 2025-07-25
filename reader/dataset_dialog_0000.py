from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import json
from utils.misc import align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
# {"text": np.array, "sentence_splits": list, "summary": np.array(summary will always be treated as one sentence)}
class XSumDataset(data.Dataset):
    
    def __init__(self, data_dir, mode, tokenizer, eos_id=50256, document_threshold=900, summary_threshold=100, seperator=" ", **kwargs):
        # data_name = data_dir + '/' + mode + '.json'
        # data_
        data_name = data_dir # it should be 0000 - 1111
        self._tokenizer = tokenizer
        self._eos_id = eos_id
        self.document_threshold = document_threshold 
        self.summary_threshold = summary_threshold
        self._lines = self.xsum_load_dataset(data_name, mode=mode, sep=seperator)
        # elif "cnn" in data_name or "gigaword" in data_name:
        #     self._lines = self.cnn_giga_load_dataset(data_name, mode=mode, sep=seperator)
        # else:
        #     raise Exception('invalid dataset name')
    
    def get_examples(self, datapath):
        with open(datapath, "r") as json_file:
            examples = json.load(json_file)
        return examples

    def get_input_ids(self, input_id_file):
        with open(input_id_file, "r") as f:
            input_id_data = f.readlines()
            input_ids = []
            for line in tqdm(input_id_data):
                line = line.strip().split(" ")
                line = self._tokenizer.convert_tokens_to_ids(line)
                input_ids.append(line)
        return input_ids
    
    def get_input_parses(self, input_parse_file):
        with open(input_parse_file, "r") as f:
            input_parse_data = f.readlines()
            input_parses = []
            for line in tqdm(input_parse_data):
                line = eval(line.strip())
                input_parses.append(line)
        return input_parses


    def xsum_load_dataset(self, data_path_or_dir, **kwargs) -> List[Dict]:
        print(data_path_or_dir)
        self.mode = kwargs.pop('mode')
        
        seperator = None if 'sep' not in kwargs else kwargs['sep']
        if self.mode == "train":
            self.input_examples = self.get_examples('dialog/dailydialog_train.json')
            input_id_file = "dialog/dailydialog_train_0000.txt"
            input_parse_file = "dialog/dailydialog_train_0000_parses.txt"
        elif self.mode == "test":
            self.input_examples = self.get_examples('dialog/dailydialog_test.json')
            input_id_file = "dialog/dailydialog_test_0000.txt"
            input_parse_file = "dialog/dailydialog_test_0000_parses.txt"
        
        self.input_ids = self.get_input_ids(input_id_file)
        self.input_parses = self.get_input_parses(input_parse_file)
        # print("input_ids: ", self.input_ids[0:3])
        # print("input_parses: ", self.input_parses[0:3])
        # exit()
        input_items = []
        sent_idx = 0
        for input_example in tqdm(self.input_examples):
            text = []
            summarytext = []
            summary_for_train_flag = True
            sentence_splits = [0]
            sentence_parses = []
            doc_full = False
            for document_sent in input_example["document"]:
                if doc_full:
                    sent_idx += 1
                    continue
                if document_sent == "__eou__":
                    ids = [self._eos_id]
                    parses = []
                else:
                    ids = self.input_ids[sent_idx]
                    parses = self.input_parses[sent_idx]
                if sentence_splits[-1] + len(ids) <= self.document_threshold:
                    text = text + ids
                    sentence_splits.append(len(text))
                    sentence_parses.append(parses)
                    assert len(ids) == len(parses) + 1
                else:
                    doc_full = True
                sent_idx += 1
            # text = text + [self._tokenizer.convert_tokens_to_ids("Summary"), self._tokenizer.convert_tokens_to_ids(":")]
            # # if self.mode == "train" or self.mode == "train_tiny":
            # sentence_splits.append(len(text))
            # sentence_parses.append([0]) 

            for summary_sent in input_example["summary"]:
                ids = self.input_ids[sent_idx]
                parses = self.input_parses[sent_idx]
                summarytext = summarytext + ids
                if self.mode == "train" or self.mode == "train_tiny":
                    text = text + ids
                    sentence_splits.append(len(text))
                    sentence_parses.append(parses)
                    assert len(ids) == len(parses) + 1
                sent_idx += 1
                # if summary_for_train_flag and (self.mode == "train" or self.mode == "train_tiny"):
                #     if len(summarytext) <= self.summary_threshold:
                #         text = text + ids
                #     else:
                #         sp = self.summary_threshold - len(summarytext)
                #         text = text + ids[:sp]
                #         summary_for_train_flag = False
                
            if self.mode == "train" or self.mode == "train_tiny":
                text = text + [self._eos_id]
                sentence_splits.append(len(text))
                sentence_parses.append([])
            text = np.array(text)
            summarytext = np.array(summarytext)
            current_item = {"text": text, "sentence_splits": sentence_splits[1:], "sentence_parses": sentence_parses, "summary": summarytext}
            input_items.append(current_item)
        assert sent_idx == len(self.input_ids)
        return input_items


    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)
