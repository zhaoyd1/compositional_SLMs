from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
from masking_1011 import utils as masking_utils
from masking_1011 import masking_types as types
from typing import Optional
import torch

class GPT2Dataset(data.Dataset):
    
    def __init__(self, ds,
                 max_seq_len=2048,
                 num_samples=None,
                 weighted=True,
                 tg=False,
                 bos_id=50256,
                 ranges: Optional[masking_utils.TokenTypeRanges]=None, 
                 mask_rules=None,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.tg = tg
        self.ranges = ranges
        self.mask_rules = mask_rules
        self.num_samples = num_samples
        self.bos_id = bos_id
        if num_samples is None:
            self.num_samples = 20 * self.ds_len
        self.max_seq_len = max_seq_len
        self.weighted = weighted
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            print(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        orig_data_idx = data_idx
        tokens, splits, transformed_splits, spans = self.getidx(data_idx)
        num_tokens = splits[-1]
        sentence_splits = []
        sentence_transformed_splits = []
        sentence_spans = []

        if len(splits) > 1:
        # tokens_to_strip = num_tokens - self.max_seq_len
        # if tokens_to_strip > 0:
            tokens_to_strip = splits[-2]
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            start_idx = 0
            for split in splits:
                if split >= strip_left_tokens:
                    break
                start_idx += 1
            
            strip_left_tokens = splits[start_idx]
            transformed_strip_left_tokens = transformed_splits[start_idx]
            tokens = tokens[strip_left_tokens:]
            spans = spans[start_idx + 1:]
            splits = splits[start_idx + 1:]
            transformed_splits = transformed_splits[start_idx + 1:]
            
            full = False 
            for i in range(len(splits)):
                if splits[i] - strip_left_tokens <= self.max_seq_len:
                    sentence_splits.append(splits[i] - strip_left_tokens)
                    sentence_transformed_splits.append(transformed_splits[i] - transformed_strip_left_tokens)
                    sentence_spans.append(spans[i])
                else:
                    full = True
                    break

            if len(sentence_splits) != 0:
                tokens = tokens[:sentence_splits[-1]]
                length = sentence_splits[-1]
                transformed_length = sentence_transformed_splits[-1]
            else:
                print(full)
                length = 0
                transformed_length = 0
                tokens = []
            
            if not full:
                while length < self.max_seq_len:
                    data_idx = (data_idx + 1) % self.ds_len
                    new_tokens, splits, new_transformed_splits, new_spans = self.getidx(data_idx)
                    for i in range(len(splits)):
                        if splits[i] + length <= self.max_seq_len:
                            sentence_splits.append(splits[i] + length)
                            sentence_transformed_splits.append(new_transformed_splits[i] + transformed_length)
                            sentence_spans.append(new_spans[i])
                        else:
                            full = True
                            break
                    length = sentence_splits[-1]
                    transformed_length = sentence_transformed_splits[-1]
                    tokens = np.concatenate([tokens, new_tokens], axis=0)
                    tokens = tokens[:length]
                    if full:
                        break

            # print(splits[start_idx:])
            # print(transformed_splits[start_idx:])
            # print(one_sent_length)
            # exit()
        elif num_tokens <= self.max_seq_len:
            length = num_tokens
            for i in range(len(splits)):
                sentence_splits.append(splits[i])
                sentence_transformed_splits.append(transformed_splits[i])
                sentence_spans.append(spans[i])
            
            transformed_length = sentence_transformed_splits[-1]

            assert length == sentence_splits[-1]
            full = False
            
            while length < self.max_seq_len:
                data_idx = (data_idx + 1) % self.ds_len
                new_tokens, splits, new_transformed_splits, new_spans = self.getidx(data_idx)
                for i in range(len(splits)):
                    if splits[i] + length <= self.max_seq_len:
                        sentence_splits.append(splits[i] + length)
                        sentence_transformed_splits.append(new_transformed_splits[i] + transformed_length)
                        sentence_spans.append(new_spans[i])
                    else:
                        full = True
                        break
                length = sentence_splits[-1]
                transformed_length = sentence_transformed_splits[-1]
                tokens = np.concatenate([tokens, new_tokens], axis=0)
                tokens = tokens[:length]
                if full:
                    break
        
        else:
            tokens = []
            length = 0
            transformed_length = 0
            full = False
            while length < self.max_seq_len:
                data_idx = (data_idx + 1) % self.ds_len
                new_tokens, splits, new_transformed_splits, new_spans = self.getidx(data_idx)
                for i in range(len(splits)):
                    if splits[i] + length <= self.max_seq_len:
                        sentence_splits.append(splits[i] + length)
                        sentence_transformed_splits.append(new_transformed_splits[i] + transformed_length)
                        sentence_spans.append(new_spans[i])
                    else:
                        full = True
                        break
                length = sentence_splits[-1]
                transformed_length = sentence_transformed_splits[-1]
                tokens = np.concatenate([tokens, new_tokens], axis=0)
                tokens = tokens[:length]
                if full:
                    break

        assert len(sentence_transformed_splits) == len(sentence_splits)
        assert len(sentence_transformed_splits) == len(sentence_spans)

        def span_update(span, offset):
            new_spans = {}
            for key, value in span.items():
                new_spans[key] = []
                for span_item in value:
                    tmp_span = []
                    for i in range(len(span_item)):
                        # print(span_item[i])
                        tmp_span.append(span_item[i] + offset)
                    new_spans[key].append(tmp_span)
            return new_spans
        # print(sentence_spans)
        # print(sentence_splits)
        # print(sentence_transformed_splits)
        # exit()
        new_sentence_spans = []
        for i in range(0, len(sentence_spans)):
            if i == 0:
                new_sentence_spans.append(span_update(sentence_spans[i], 0))
            else:
                new_sentence_spans.append(span_update(sentence_spans[i], sentence_transformed_splits[i - 1]))
        
        sentence_spans = new_sentence_spans
        # print(sentence_transformed_splits)
        def get_max(span):
            max_ = 0
            for key, value in span.items():
                for span_item in value:
                    max_ = max(max_, max(span_item))
            return max_
        
        # if len(sentence_spans) == 0:
        #     print(sentence_spans)
        #     print(sentence_splits)
        #     print(sentence_transformed_splits)
        #     tokens, splits, transformed_splits, spans = self.getidx(orig_data_idx)
        #     print(spans[3])
        #     print(splits)
        #     print(transformed_splits)
        # if get_max(sentence_spans[-1]) + 2 != sentence_transformed_splits[-1]:
        #     print(get_max(sentence_spans[-1]) + 2)
        #     print(sentence_transformed_splits[-1])
        #     print(sentence_spans)
        #     print(sentence_splits)
        #     print(sentence_transformed_splits)
        #     exit()
        # assert get_max(sentence_spans[-1]) + 2 == sentence_transformed_splits[-1]

        merged_spans = {}
        for i in range(len(sentence_spans)):
            for key, value in sentence_spans[i].items():
                if key not in merged_spans:
                    merged_spans[key] = []
                merged_spans[key].extend(value)
        del sentence_spans
        # if transformed_length > self.max_seq_len:
        #     transformed_length = self.max_seq_len
        length = self.max_seq_len
        src_ = torch.LongTensor([self.bos_id] + tokens.tolist()) 
        tgt_ = torch.LongTensor(tokens.tolist() + [self.ranges.pad_token])
        info_tuple = masking_utils.compute_token_types(
            {"inputs": src_, "labels": tgt_}, self.ranges
        )
        chunks = self.mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                                                    info_tuple['labels'], info_tuple['labels_ttypes'])
        
        chunks = [types.Chunk(None, *chunk) for chunk in chunks]
        # print(self.max_seq_len)
        # print(len(chunks[0].inputs))
        assert len(chunks[0].inputs) == self.max_seq_len + 1
        chunk = chunks[0]
        src_p = chunk.inputs[:length + 1]
        tgt_p = chunk.labels[:length + 1]
        mask = chunk.attn_mask[:length + 1, :length + 1]
        position_ids = chunk.position_ids[:length + 1]
        for i in range(len(mask)):
            mask[i, i] = 1
        chunk_len = len(chunk.inputs)
        relpos = chunk.attn_relpos[:length + 1, chunk_len:chunk_len + length + 1]

        return {"src": np.array(src_p), "tgt": np.array(tgt_p), "mask": np.array(mask), "relpos": np.array(relpos), 
                "transformed_splits": np.array(sentence_transformed_splits),
                "merged_spans": merged_spans, "position_ids": np.array(position_ids)}


    def getidx(self, data_idx):
        token_ids, splits, transformed_splits, spans = self.ds[data_idx]
        if len(splits) == 0:
            splits = [len(token_ids)]
        #     splits.append(len(token_ids))
        return token_ids, splits, transformed_splits, spans