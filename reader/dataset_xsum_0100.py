from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import json
from utils.misc import align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from typing import Dict, List
import torch
from masking_gpst import utils as masking_utils
from masking_gpst import masking_types as types
from tqdm import tqdm
# {"text": np.array, "sentence_splits": list, "summary": np.array(summary will always be treated as one sentence)}
class XSumDataset(data.Dataset):
    
    def __init__(self, data_dir, mode, tokenizer, eos_id=50256, document_threshold=900, summary_threshold=100, max_len=1024, seperator=" ", ranges=None, maskrules=None, **kwargs):
        if mode == 'train' or mode == 'train_tiny':
            data_name = 'train/xsum_train_0001.txt'
        elif mode == 'test' or mode == 'test_tiny':
            data_name = 'test/xsum_test_0001.txt'
        self._tokenizer = tokenizer
        self.ranges = ranges
        self.bos_id = ranges.start_token
        self.opening_nt = ranges.opening_non_terminals
        self.closing_nt = ranges.closing_non_terminals
        self.pad_id = ranges.pad_token
        self.max_len = max_len
        self.maskrules = maskrules
        self._eos_id = eos_id
        self.document_threshold = document_threshold 
        self.summary_threshold = summary_threshold
        self._lines = self.xsum_load_dataset(data_name, mode=mode, sep=seperator)
    
    def get_examples(self, datapath):
        with open(datapath, "r") as json_file:
            examples = json.load(json_file)
        return examples

    def _to_ids_and_atom_spans(self, text, seperator):
        if seperator is None:
            tokens = self._tokenizer.tokenize(text)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            atom_spans = None
            indices_mapping = None
        else:
            sentence, spans = get_sentence_from_words(text.strip().split(seperator), seperator)
            outputs = self._tokenizer.encode_plus(sentence,
                                                  add_special_tokens=False,
                                                  return_offsets_mapping=True)
            new_spans = outputs['offset_mapping']
            word_starts, word_ends = align_spans(spans, new_spans)
            atom_spans = []
            indices_mapping = []
            for st, ed in zip(word_starts, word_ends):
                if st != ed:
                    atom_spans.append((st, ed))
                indices_mapping.append([st, ed])
            ids = outputs['input_ids']
            atom_spans = atom_spans
        return ids, atom_spans, indices_mapping

    def xsum_load_dataset(self, data_path_or_dir, **kwargs) -> List[Dict]:
        print(data_path_or_dir)
        self.mode = kwargs.pop('mode')
        if self.mode == 'train' or self.mode == 'train_tiny':
            data_path = 'train/train_clean.json'
        elif self.mode == 'test' or self.mode == 'test_tiny':
            data_path = 'test/test_clean.json'
        seperator = None if 'sep' not in kwargs else kwargs['sep']
        self.input_examples = self.get_examples(data_path)
        self.input_sequences = []
        with open(data_path_or_dir, "r") as file:
            for line in tqdm(file):
                cur_seq = self._tokenizer.convert_tokens_to_ids(line.strip().split())
                self.input_sequences.append(cur_seq)
        input_items = []
        seq_idx = 0
        for input_example in tqdm(self.input_examples):
            text = []
            summarytext = []
            summary_for_train_flag = True
            sentence_splits = [0]
            transformed_sentence_splits = [0]
            transformed_len = 0
            length = 0
            full = False
            for document_sent in input_example["document"]:
                if full:
                    seq_idx += 1
                    continue
                ids = self.input_sequences[seq_idx]
                cur_len = len(ids)
                cur_transformed_len = len(ids) + sum([1 for i in range(len(ids)) if ids[i] >= self.closing_nt[0] and ids[i] < self.closing_nt[1]])
                seq_idx += 1
                if transformed_sentence_splits[-1] + cur_transformed_len <= self.document_threshold:
                    text = text + ids
                    sentence_splits.append(len(text))
                    transformed_sentence_splits.append(transformed_len + cur_transformed_len)
                    transformed_len += cur_transformed_len
                    length += cur_len
                    assert length == len(text)
                else:
                    full = True
                    continue
                if sentence_splits[-1] >= self.document_threshold:
                    full = True
                    continue
            
            text = text + [self._tokenizer.convert_tokens_to_ids("Summary"), 
                           self._tokenizer.convert_tokens_to_ids(":"),
                           self._tokenizer.convert_tokens_to_ids("X)")]
            sentence_splits.append(len(text))
            transformed_sentence_splits.append(transformed_len + 4)
            transformed_len += 4
            length += 3
            # if self.mode == "train" or self.mode == "train_tiny":
            #     sentence_splits.append(len(text))
            summary_splits = [0]
            summary_transformed_splits = [0]
            summary_transformed_len = 0
            summary_len = 0
            full = False
            for summary_sent in input_example["summary"]:
                if full:
                    seq_idx += 1
                    continue
                ids = self.input_sequences[seq_idx]
                summarytext = summarytext + ids
                cur_len = len(ids)
                seq_idx += 1
                cur_transformed_len = len(ids) + sum([1 for i in range(len(ids)) if ids[i] >= self.closing_nt[0] and ids[i] < self.closing_nt[1]])
                if summary_for_train_flag and (self.mode == "train" or self.mode == "train_tiny"):
                    if summary_transformed_splits[-1] + cur_transformed_len <= self.summary_threshold:
                        text = text + ids
                        summary_splits.append(len(summarytext))
                        sentence_splits.append(len(text))
                        summary_transformed_splits.append(summary_transformed_len + cur_transformed_len)
                        transformed_sentence_splits.append(transformed_len + cur_transformed_len)
                        summary_transformed_len += cur_transformed_len
                        summary_len += cur_len
                        transformed_len += cur_transformed_len
                        length += cur_len
                        assert summary_len == len(summarytext)
                        assert length == len(text)
                    else:
                        full = True
                        continue

                           
            if self.mode == "train" or self.mode == "train_tiny":
                text = text + [self._eos_id]
                length += 1
                transformed_len += 1
                sentence_splits.append(len(text))
                transformed_sentence_splits.append(transformed_len)
            
            text = np.array(text)
            # print(text)
            summarytext = np.array(summarytext)
            assert transformed_len <= self.max_len
            
            # chunk_len = len(chunk.inputs)
            # relpos = chunk.attn_relpos[:self.max_len, chunk_len:chunk_len + self.max_len]
            if self.mode == "test" or self.mode == "test_tiny":
                current_item = {"text": text, "summary": summarytext, "transformed_len": transformed_len}
            else:
                current_item = {"text": text, "summary": summarytext}
            # current_item = {"src": np.array(src_p), "tgt": np.array(tgt_p), "mask": np.array(mask), "relpos": None, "summary": summarytext}
            input_items.append(current_item)
        assert seq_idx == len(self.input_sequences)
        print("seq_idx: ", seq_idx)
        print("len of input_sequences: ", len(self.input_sequences))
        return input_items

    # def cnn_giga_load_dataset(self, data_path_or_dir, **kwargs) -> List[Dict]:
    #     print(data_path_or_dir)
    #     self.mode = kwargs.pop('mode')

    #     seperator = None if 'sep' not in kwargs else kwargs['sep']
    #     self.input_examples = self.get_examples(data_path_or_dir)
    #     input_items = []
    #     for input_example in self.input_examples:
    #         text = []
    #         summarytext = []
    #         summary_for_train_flag = True
    #         sentence_splits = [0]
    #         for document_sent in input_example["document"]:
    #             ids, atom_spans, indices_mapping = \
    #                 self._to_ids_and_atom_spans(document_sent, seperator)
    #             if sentence_splits[-1] + len(ids) <= self.document_threshold:
    #                 text = text + ids
    #                 sentence_splits.append(len(text))
    #             else:
    #                 sp = self.document_threshold - sentence_splits[-1]
    #                 text = text + ids[:sp]
    #                 sentence_splits.append(len(text))
    #             if sentence_splits[-1] >= self.document_threshold:
    #                 break
    #         text = text + [self._tokenizer.convert_tokens_to_ids("Summary"), self._tokenizer.convert_tokens_to_ids(":")]
    #         if self.mode == "train" or self.mode == "train_tiny":
    #             sentence_splits.append(len(text))
    #         for summary_sent in input_example["summary"]:
    #             ids, atom_spans, indices_mapping = \
    #                 self._to_ids_and_atom_spans(summary_sent, seperator)
    #             summarytext = summarytext + ids
    #             if summary_for_train_flag and (self.mode == "train" or self.mode == "train_tiny"):
    #                 if len(summarytext) <= self.summary_threshold:
    #                     text = text + ids
    #                     sentence_splits.append(len(text))
    #                 else:
    #                     sp = self.summary_threshold - len(summarytext)
    #                     text = text + ids[:sp]
    #                     sentence_splits.append(len(text))
    #                     summary_for_train_flag = False
    #         if self.mode == "train" or self.mode == "train_tiny":
    #             text = text + [self._eos_id]

    #         text = np.array(text)
    #         summarytext = np.array(summarytext)
    #         current_item = {"text": text, "sentence_splits": sentence_splits[1:], "summary": summarytext}
    #         input_items.append(current_item)
    #     return input_items

    def __getitem__(self, idx):
        text = self._lines[idx]["text"]
        # print(text)
        summarytext = self._lines[idx]["summary"]
        # print(summarytext)
        src_ = torch.LongTensor([self.bos_id] + text.tolist())
        tgt_ = torch.LongTensor(text.tolist() + [self.pad_id])
        info_tuple = masking_utils.compute_token_types(
            {"inputs": src_, "labels": tgt_}, self.ranges
        ) 
        chunks = self.maskrules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                                                    info_tuple['labels'], info_tuple['labels_ttypes'])
        chunks = [types.Chunk(None, *chunk) for chunk in chunks]
        assert len(chunks[0].inputs) == self.max_len
        chunk = chunks[0]
        src_p = chunk.inputs[:self.max_len]
        tgt_p = chunk.labels[:self.max_len]
        mask = chunk.attn_mask[:self.max_len, :self.max_len]
        for i in range(len(mask)):
            mask[i][i] = 1
        current_item = {"src": np.array(src_p), "tgt": np.array(tgt_p), "mask": np.array(mask), "relpos": None, "summary": summarytext}
        return current_item

    def __len__(self):
        return len(self._lines)
