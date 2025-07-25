from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
from masking import utils as masking_utils
from masking import masking_types as types
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
        if self.tg:
            tokens, splits, transformed_splits = self.getidx(data_idx)
            num_tokens = transformed_splits[-1]
            one_sent_length = transformed_splits[0]
            
            sentence_splits = []
            sentence_transformed_splits = []
            if len(transformed_splits) > 1:
            # tokens_to_strip = num_tokens - self.max_seq_len
            # if tokens_to_strip > 0:
                tokens_to_strip = transformed_splits[-2]
                strip_left_tokens = rng.randint(tokens_to_strip + 1)
                start_idx = 0
                for split in transformed_splits:
                    if split >= strip_left_tokens:
                        break
                    start_idx += 1
                
                strip_left_tokens = splits[start_idx]
                tokens = tokens[strip_left_tokens:]
                transformed_length = transformed_splits[-1] - transformed_splits[start_idx]
                one_sent_length = transformed_splits[start_idx + 1] - transformed_splits[start_idx]
                
                while transformed_length < self.max_seq_len:
                    data_idx = (data_idx + 1) % self.ds_len
                    new_tokens, splits, transformed_splits = self.getidx(data_idx)
                    new_len = transformed_splits[-1]
                    if new_len + transformed_length <= self.max_seq_len:
                        transformed_length += new_len
                        tokens = np.concatenate([tokens, new_tokens], axis=0)
                    else:
                        break
                # print(splits[start_idx:])
                # print(transformed_splits[start_idx:])
                # print(one_sent_length)
                # exit()
            elif num_tokens <= self.max_seq_len:
                transformed_length = transformed_splits[-1]
                while transformed_length < self.max_seq_len:
                    data_idx = (data_idx + 1) % self.ds_len
                    new_tokens, splits, transformed_splits = self.getidx(data_idx)
                    new_len = transformed_splits[-1]
                    if new_len + transformed_length <= self.max_seq_len:
                        transformed_length += new_len
                        tokens = np.concatenate([tokens, new_tokens], axis=0)
                    else:
                        break
            
            else:
                tokens = []
                transformed_length = 0
                while transformed_length < self.max_seq_len:
                    data_idx = (data_idx + 1) % self.ds_len
                    new_tokens, splits, transformed_splits = self.getidx(data_idx)
                    new_len = transformed_splits[-1]
                    if new_len + transformed_length <= self.max_seq_len:
                        transformed_length += new_len
                        tokens = np.concatenate([tokens, new_tokens], axis=0)
                    else:
                        break
            # if transformed_length > self.max_seq_len:
            #     transformed_length = self.max_seq_len
            transformed_length = self.max_seq_len
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
            src_p = chunk.inputs[:transformed_length + 1]
            tgt_p = chunk.labels[:transformed_length + 1]
            mask = chunk.attn_mask[:transformed_length + 1, :transformed_length + 1]
            for i in range(len(mask)):
                mask[i, i] = 1
            chunk_len = len(chunk.inputs)
            relpos = chunk.attn_relpos[:transformed_length + 1, chunk_len:chunk_len + transformed_length + 1]

            return {"src": np.array(src_p), "tgt": np.array(tgt_p), "mask": np.array(mask), "relpos": np.array(relpos)}

        else:
            tokens, splits, parses = self.getidx(data_idx)
            # truncate or pad tokens
            num_tokens = len(tokens)
            
            # print("num_tokens: ", num_tokens, "max_seq_len: ", self.max_seq_len, "tokens_to_strip: ", tokens_to_strip)
            sentence_splits = []
            sentence_parses = []
            # randomly choose a position for start
            
            if len(splits) > 1:
                tokens_to_strip = splits[-2]
            # tokens_to_strip = num_tokens - self.max_seq_len
            # if tokens_to_strip > 0:
                strip_left_tokens = rng.randint(tokens_to_strip + 1)
                start_idx = 0
                for split in splits:
                    if split >= strip_left_tokens:
                        break
                    start_idx += 1
                
                # start from a complete sentence
                strip_left_tokens = splits[start_idx]
                tokens = tokens[strip_left_tokens:]

                for split, parse in zip(splits, parses):
                    if split > strip_left_tokens and split - strip_left_tokens <= self.max_seq_len:
                        sentence_splits.append(split - strip_left_tokens)
                        sentence_parses.append(parse)
                    elif split > strip_left_tokens:
                        break    
                # tokens = tokens[:sentence_splits[-1]]
                strip_right_tokens = len(tokens) - self.max_seq_len
                if strip_right_tokens > 0:
                    tokens = tokens[:-strip_right_tokens]

            elif len(tokens) <= self.max_seq_len:
                sentence_splits = [s for s in splits]
                sentence_parses = [p for p in parses]
            else:
                tokens = []

            # if strip_right_tokens >= 0 or len(tokens) == self.max_seq_len:
            #     return {'text': np.array(tokens),  "sentence_splits": sentence_splits, "sentence_parses": sentence_parses}

            while (len(tokens) < self.max_seq_len):
                # if self.random_across_doc_sampling:
                #     data_idx = self.get_weighted_samples(rng)
                # else:
                data_idx = (data_idx + 1) % self.ds_len
                new_tokens, splits, parses = self.getidx(data_idx)
                assert splits[-1] <= len(new_tokens), f'{len(new_tokens)} / {splits[-1]}'
                if len(sentence_splits) > 0:
                    assert len(tokens) >= sentence_splits[-1], f'{len(tokens)}/{sentence_splits[-1]}'
                    
                for split, parse in zip(splits, parses):
                    if split + len(tokens) <= self.max_seq_len:
                        sentence_splits.append(split + len(tokens))
                        sentence_parses.append(parse)
                    else:
                        break
                # tokens += new_tokens
                tokens = np.concatenate([tokens, new_tokens], axis=0)

            tokens = tokens[:self.max_seq_len]

            return {'text': np.array(tokens),  "sentence_splits": sentence_splits, "sentence_parses": sentence_parses}

    def getidx(self, data_idx):
        if self.tg:
            token_ids, splits, transformed_splits = self.ds[data_idx]
            if len(splits) == 0:
                splits = [len(token_ids)]
            return token_ids, splits, transformed_splits
        else:
            token_ids, splits, parses = self.ds[data_idx]
            if len(splits) == 0:
                splits = [len(token_ids)]
            #     splits.append(len(token_ids))
            return token_ids, splits, parses