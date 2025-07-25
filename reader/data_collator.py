from typing import List, Dict
import torch
import numpy as np
from multiprocessing.pool import ThreadPool
from collections import OrderedDict
from utils.vocab_builder import load_span_tokenizer
from utils.r2d2_span_tokenizer import SpanTokenizingSession
from utils.misc import align_spans
import cppbackend
import codecs


def sent_collator(input_list):
    max_len = max(map(lambda x: len(x), input_list))
    padded_ids_list = []
    masks = []
    for input_ids in input_list:
        padded_len = max_len - len(input_ids)
        padded_ids = np.append(input_ids, np.array([0] * padded_len))
        padded_ids_list.append(padded_ids)
        masks.append([1] * len(input_ids) + [0] * padded_len)
        # mask = np.zeros((max_len, max_len))
        # # mask[:len(input_ids), :len(input_ids)].fill(1)
        # masks.append(mask)
    return {"input_ids": torch.tensor(padded_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)}

class DefaultCollator:
    def __init__(self):
        pass
    
    def generative_r2d2_collate_fn_0010(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":..., "mask":..., "relpos":..., "pointer_tgt_pos":..., "pointer_mask":...,
                         "transformed_splits":..., "merged_spans":..., "position_ids":...},...]
        '''
        
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        attn_masks = []
        rel_positions = []
        pointer_tgt_pos = []
        pointer_mask = []
        position_ids = []
        offset = 1 # 0 is reserved for bos
        all_merged_spans = {}

        def span_update(span, offset):
            for key, value in span.items():
                for span_item in value:
                    for i in range(len(span_item)):
                        # print(span_item[i])
                        span_item[i] += offset
        
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            mask = item['mask']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
            attn_masks.append(mask)
            pointer_tgt_pos.append(item['pointer_tgt_pos'])
            pointer_mask.append(item['pointer_mask'])
            position_ids.append(item['position_ids'])
            
            merged_spans = item['merged_spans']
            span_update(merged_spans, offset)
            offset += max_src_len
            for key, value in merged_spans.items():
                if key not in all_merged_spans:
                    all_merged_spans[key] = []
                all_merged_spans[key].extend(value)
            del merged_spans
        def pad_spans(spans, max_len):
            for span in spans:
                span.extend([-100] * (max_len - len(span)))

        all_max_lens = map(lambda x: max(map(lambda y: len(y), x)), all_merged_spans.values())
        all_merged_spans_tensor = {}
        for max_len, (key, value) in zip(all_max_lens, all_merged_spans.items()):
            pad_spans(value, max_len)
            all_merged_spans_tensor[key] = torch.tensor(np.array(value, dtype=np.int32), dtype=torch.long)
        del all_merged_spans

        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
                "chunk_attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool),
                "pointer_tgt_pos": torch.tensor(np.array(pointer_tgt_pos, dtype=np.int32), dtype=torch.long),
                "pointer_mask": torch.tensor(np.array(pointer_mask, dtype=np.int32), dtype=torch.bool),
                "position_ids": torch.tensor(np.array(position_ids, dtype=np.int32), dtype=torch.long),
                "all_merged_spans": all_merged_spans_tensor}

    def generative_r2d2_collate_fn_1010(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":..., "mask":..., "relpos":...,
                         "transformed_splits":..., "merged_spans":..., "position_ids":...},...]
        '''
        
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        attn_masks = []
        rel_positions = []
        pointer_tgt_pos = []
        pointer_mask = []
        position_ids = []
        offset = 1 # 0 is reserved for bos
        all_merged_spans = {}

        def span_update(span, offset):
            for key, value in span.items():
                for span_item in value:
                    for i in range(len(span_item)):
                        # print(span_item[i])
                        span_item[i] += offset
        
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            mask = item['mask']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
            attn_masks.append(mask)
            position_ids.append(item['position_ids'])
            
            merged_spans = item['merged_spans']
            span_update(merged_spans, offset)
            offset += max_src_len
            for key, value in merged_spans.items():
                if key not in all_merged_spans:
                    all_merged_spans[key] = []
                all_merged_spans[key].extend(value)
            del merged_spans
        def pad_spans(spans, max_len):
            for span in spans:
                span.extend([-100] * (max_len - len(span)))

        all_max_lens = map(lambda x: max(map(lambda y: len(y), x)), all_merged_spans.values())
        all_merged_spans_tensor = {}
        for max_len, (key, value) in zip(all_max_lens, all_merged_spans.items()):
            pad_spans(value, max_len)
            all_merged_spans_tensor[key] = torch.tensor(np.array(value, dtype=np.int32), dtype=torch.long)
        del all_merged_spans

        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
                "chunk_attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool),
                "position_ids": torch.tensor(np.array(position_ids, dtype=np.int32), dtype=torch.long),
                "all_merged_spans": all_merged_spans_tensor}

    def generative_r2d2_collate_fn_0110(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":..., "mask":..., "relpos":..., "pointer_tgt_pos":..., "pointer_mask":...},...]
        '''
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        attn_masks = []
        rel_positions = []
        pointer_tgt_pos = []
        pointer_mask = []
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            mask = item['mask']
            relpos = item['relpos']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
            attn_masks.append(mask)
            rel_positions.append(relpos)
            pointer_tgt_pos.append(item['pointer_tgt_pos'])
            pointer_mask.append(item['pointer_mask'])
        
        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
                "attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool),
                "rel_positions": torch.tensor(np.array(rel_positions, dtype=np.int32), dtype=torch.long),
                "pointer_tgt_pos": torch.tensor(np.array(pointer_tgt_pos, dtype=np.int32), dtype=torch.long),
                "pointer_mask": torch.tensor(np.array(pointer_mask, dtype=np.int32), dtype=torch.bool)}

    def generative_r2d2_collate_fn_0111(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":..., "mask":..., "relpos":..., "pointer_tgt_pos":..., "pointer_mask":...},...]
        '''
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        attn_masks = []
        rel_positions = []
        pointer_tgt_pos = []
        pointer_mask = []
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            mask = item['mask']
            relpos = item['relpos']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
            attn_masks.append(mask)
            rel_positions.append(relpos)
            pointer_tgt_pos.append(item['pointer_tgt_pos'])
            pointer_mask.append(item['pointer_mask'])
        
        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
                "attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool),
                "rel_positions": torch.tensor(np.array(rel_positions, dtype=np.int32), dtype=torch.long),
                "pointer_tgt_pos": torch.tensor(np.array(pointer_tgt_pos, dtype=np.int32), dtype=torch.long),
                "pointer_mask": torch.tensor(np.array(pointer_mask, dtype=np.int32), dtype=torch.bool)}

    def generative_r2d2_collate_fn_tg(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":..., "mask":..., "relpos":...},...]
        '''
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        attn_masks = []
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            mask = item['mask']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
            attn_masks.append(mask)
        
        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
                "attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool)}

    def generative_r2d2_collate_fn_0001(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "attn_mask": ..., "sentence_splits":..., "seq_sentence_splits":..., "sentence_parses":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        chunk_parses_list = []
        chunk_attn_masks = []
        # chunk_masks = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        segment_ids_list = []
        # external_dict = OrderedDict()
        # external_vocab_idx = 1  # start from 1, 0 is reserved for empty span ids
        # tokenizer_session = SpanTokenizingSession(self.span_tokenizer)
        for sent_id, item in enumerate(input_list):

            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            # chunk_masks.append(chunk_mask)
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            chunk_attn_masks.append(item['attn_mask'])
            splits = item['sentence_splits']
            # splits.append(chunk_size)
            parses = item['sentence_parses']

            prev_idx = 0
            chunk_parse = []
            start_flag = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, (split_idx, parse) in enumerate(zip(splits, parses)):
                if parse == []:
                    parse_max_index = -1
                else:
                    parse_max_index = max(parse)
                assert parse_max_index == len(parse) - 1
                if parse_max_index + 1 + prev_idx + 1 != split_idx:
                    print(parse_max_index + 1 + prev_idx + 1, split_idx)
                    print(parse)
                    print(split_idx, prev_idx)
                    exit()
                assert parse_max_index + 1 + prev_idx + 1 == split_idx # index starts from 0 for +1,  n split points for n + 1 tokens

                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    # if self.span_tokenizer is not None:
                    #     span_idx = tokenizer_session.tokenize(ids_segment)
                    #     span_indices.append(span_idx)
                    if chunk_parse == [] and start_flag == 0:
                        chunk_parse += parse
                        start_flag = 1
                    else:
                        chunk_parse = [len(chunk_parse)] + chunk_parse
                        accum_parse = [parse_id + len(chunk_parse) for parse_id in parse]
                        chunk_parse += accum_parse

                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx
        
            chunk_ids_list.append(item['text'][:prev_idx])
            chunk_parses_list.append(chunk_parse)
        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        # padding

        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))
        
        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            if pad_len > 0:
                chunk_parses_list[chunk_id] = chunk_parses_list[chunk_id] + [p for p in range(max(len(chunk_ids)-1, 0), max_input_len-1)]

        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "chunk_parses": torch.tensor(np.array(chunk_parses_list, dtype=np.int32), dtype=torch.long),
                "chunk_attn_masks": torch.tensor(np.array(chunk_attn_masks, dtype=np.int32), dtype=torch.long)}

    def generative_gpt2_fn(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":... },...]
        '''
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        chunk_masks = []
        for item in input_list:
            src = item['src']
            masks = [1] * len(src) + [0] * (max_src_len - len(src))
            if len(src) < max_src_len:
                src = np.append(src, np.array([-100] * (max_src_len - len(src))))
            input_ids.append(src)
            chunk_masks.append(masks)

        
        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(chunk_masks, dtype=np.int32), dtype=torch.long)}

    def generative_gpt2_fn_tg_seq(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list:[{"src":..., "tgt":... },...]
        '''
        max_src_len = max(map(lambda x: len(x['src']), input_list))
        input_ids = []
        tgt_ids = []
        for item in input_list:
            src = item['src']
            tgt = item['tgt']
            assert len(src) == max_src_len
            input_ids.append(src)
            tgt_ids.append(tgt)
        
        return {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
                "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long)}

    def generative_r2d2_collate_fn_ext(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "sentence_splits":..., "sentence_parses":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        chunk_parses_list = []

        # chunk_masks = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        segment_ids_list = []
        # external_dict = OrderedDict()
        # external_vocab_idx = 1  # start from 1, 0 is reserved for empty span ids
        # tokenizer_session = SpanTokenizingSession(self.span_tokenizer)
        for sent_id, item in enumerate(input_list):

            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            # chunk_masks.append(chunk_mask)
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            splits = item['sentence_splits']
            # splits.append(chunk_size)
            parses = item['sentence_parses']

            prev_idx = 0
            chunk_parse = []
            start_flag = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, (split_idx, parse) in enumerate(zip(splits, parses)):
                if parse == []:
                    parse_max_index = -1
                else:
                    parse_max_index = max(parse)
                assert parse_max_index == len(parse) - 1
                if parse_max_index + 1 + prev_idx + 1 != split_idx:
                    print(parse_max_index + 1 + prev_idx + 1, split_idx)
                    print(parse)
                    print(split_idx, prev_idx)
                    exit()
                assert parse_max_index + 1 + prev_idx + 1 == split_idx # index starts from 0 for +1,  n split points for n + 1 tokens

                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    # if self.span_tokenizer is not None:
                    #     span_idx = tokenizer_session.tokenize(ids_segment)
                    #     span_indices.append(span_idx)
                    if chunk_parse == [] and start_flag == 0:
                        chunk_parse += parse
                        start_flag = 1
                    else:
                        chunk_parse = [len(chunk_parse)] + chunk_parse
                        accum_parse = [parse_id + len(chunk_parse) for parse_id in parse]
                        chunk_parse += accum_parse

                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx
        
            chunk_ids_list.append(item['text'][:prev_idx])
            chunk_parses_list.append(chunk_parse)
        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        # padding

        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))
        
        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            if pad_len > 0:
                chunk_parses_list[chunk_id] = chunk_parses_list[chunk_id] + [p for p in range(max(len(chunk_ids)-1, 0), max_input_len-1)]

        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "chunk_parses": torch.tensor(np.array(chunk_parses_list, dtype=np.int32), dtype=torch.long)}


class GlueCollator(DefaultCollator):
    def __init__(self, clstgt_ids, padding=-1):
        self._clstgt_ids = clstgt_ids
        self._padding = padding
        super().__init__()

    def generative_r2d2_glue_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        if self._padding != -1:
            for item in input_list:
                appen = np.array([0]*(self._padding - len(item['text'])))
                item['text'] = np.concatenate((item['text'], appen))
        origin_dict = self.generative_r2d2_collate_fn(input_list)
        eos_labels = []
        for item in input_list:
            eos_labels.append(self._clstgt_ids[item["label"]])
        origin_dict["eos_labels"] = np.array(eos_labels)
        return origin_dict

class XSumCollator(DefaultCollator):
    def __init__(self, mode, padding=-1):
        self.mode = mode
        self._padding = padding
        super().__init__()
    
    def generative_r2d2_xsum_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        if self._padding != -1:
            for item in input_list:
                appen = np.array([0]*(self._padding - len(item['text'])))
                item['text'] = np.concatenate((item['text'], appen))
        origin_dict = self.generative_r2d2_collate_fn_ext(input_list)
        # print("origin_dict: ", origin_dict)
        # # new added
        if self.mode == "test" or self.mode == "valid":
            chunk_summarys_list = []
            max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            for sent_id, item in enumerate(input_list):
                chunk_summarys_list.append(item['summary'])
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None
        
        return origin_dict
    
    def xsum_collate_fn_1111(self, input_list) -> Dict[str, torch.Tensor]:
        origin_dict = self.generative_r2d2_collate_fn_tg(input_list)
        # new added
        if self.mode == "test" or self.mode == "valid" or self.mode == "test_tiny":
            chunk_summarys_list = []
            # max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            max_summary_len = 0
            for sent_id, item in enumerate(input_list):
                new_item = [item['summary'][i] for i in range(len(item['summary'])) if item['summary'][i] < 50289]
                chunk_summarys_list.append(new_item)
                if len(new_item) > max_summary_len:
                    max_summary_len = len(new_item)
                
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None

        return origin_dict
    
    def xsum_collate_fn_0110(self, input_list) -> Dict[str, torch.Tensor]:
        origin_dict = self.generative_r2d2_collate_fn_0110(input_list)
        # new added
        if self.mode == "test" or self.mode == "valid" or self.mode == "test_tiny":
            chunk_summarys_list = []
            # max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            max_summary_len = 0
            for sent_id, item in enumerate(input_list):
                new_item = [item['summary'][i] for i in range(len(item['summary'])) if item['summary'][i] < 50289]
                chunk_summarys_list.append(new_item)
                if len(new_item) > max_summary_len:
                    max_summary_len = len(new_item)
                
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None

        return origin_dict

    def xsum_collate_fn_0111(self, input_list) -> Dict[str, torch.Tensor]:
        origin_dict = self.generative_r2d2_collate_fn_0111(input_list)
        # new added
        if self.mode == "test" or self.mode == "valid" or self.mode == "test_tiny":
            chunk_summarys_list = []
            # max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            max_summary_len = 0
            for sent_id, item in enumerate(input_list):
                new_item = [item['summary'][i] for i in range(len(item['summary'])) if item['summary'][i] < 50289]
                chunk_summarys_list.append(new_item)
                if len(new_item) > max_summary_len:
                    max_summary_len = len(new_item)
                
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None

        return origin_dict

    def xsum_collator_fn_gpt2(self, input_list) -> Dict[str, torch.Tensor]:
        origin_dict = self.generative_gpt2_fn(input_list)
        # print("origin_dict: ", origin_dict)
        # exit()
        # new added
        if self.mode == "test" or self.mode == "valid" or self.mode == "test_tiny":
            chunk_summarys_list = []
            # max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            max_summary_len = 0
            for sent_id, item in enumerate(input_list):
                new_item = [item['summary'][i] for i in range(len(item['summary'])) if item['summary'][i] < 50289]
                chunk_summarys_list.append(new_item)
                if len(new_item) > max_summary_len:
                    max_summary_len = len(new_item)
                
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None

        return origin_dict

    def xsum_collator_fn_gpt2_tg_seq(self, input_list) -> Dict[str, torch.Tensor]:
        origin_dict = self.generative_gpt2_fn_tg_seq(input_list)
        # new added
        if self.mode == "test" or self.mode == "valid":
            chunk_summarys_list = []
            max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            for sent_id, item in enumerate(input_list):
                new_item = [item['summary'][i] for i in range(len(item['summary'])) if item['summary'][i] < 50289]
                chunk_summarys_list.append(new_item)
                if len(new_item) > max_summary_len:
                    max_summary_len = len(new_item)
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None

        return origin_dict

    def xsum_collate_fn_0001(self, input_list) -> Dict[str, torch.Tensor]:
        if self._padding != -1:
            for item in input_list:
                appen = np.array([0]*(self._padding - len(item['text'])))
                item['text'] = np.concatenate((item['text'], appen))
        origin_dict = self.generative_r2d2_collate_fn_0001(input_list)
        # print("origin_dict: ", origin_dict)
        # # new added
        if self.mode == "test" or self.mode == "valid":
            chunk_summarys_list = []
            max_summary_len = max(map(lambda x: len(x['summary']), input_list))
            for sent_id, item in enumerate(input_list):
                chunk_summarys_list.append(item['summary'])
            for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
                pad_len = max_summary_len - len(chunk_ids)
                chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
            origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        else:
            origin_dict["summarys"] = None
        
        return origin_dict
    
class TextCollator(DefaultCollator):
    def __init__(self, tokenizer, splitter, external_vocab_path=None):
        self.tokenizer = tokenizer
        self.splitter = splitter
        super().__init__(external_vocab_path=external_vocab_path)

    def collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        atom_spans_batch = []
        for sentence in input_list:
            tokens, split_word = self.splitter(sentence)
            offset = 0
            spans = []
            for word in tokens:
                length = len(word)
                spans.append((offset, offset + length))
                offset += length + len(split_word)
            outputs = self.tokenizer.encode_plus(sentence,
                                                 add_special_tokens=False,
                                                 return_offsets_mapping=True)
            input_ids = outputs['input_ids']
            offset_mapping = outputs['offset_mapping']
            word_starts, word_ends = align_spans(spans, offset_mapping)
            atom_spans = [] # minimal span should be a whole word
            for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                if ed > st:
                    atom_spans.append([st, ed])
            input_ids_list.append({'text':input_ids, 'sentence_splits': []})
            atom_spans_batch.append(atom_spans)

        out_dict = self.generative_r2d2_collate_fn(input_ids_list)
        out_dict['atom_spans'] = atom_spans_batch
        return out_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from dataset import GPT2Dataset
    from tqdm import tqdm
    from reader.lazy_loader import LazyLoader
    tokenizer = AutoTokenizer.from_pretrained("/home/zhaoyd/supervised_gpst/data/gpt2-small")

    bllip_lazy_loader = LazyLoader("/home/zhaoyd/supervised_gpst/corpus/bllip_train", is_array=True)
    bllip_dataset = GPT2Dataset(bllip_lazy_loader, num_samples=200, max_seq_len=1024)
    print(len(bllip_dataset))
    print("dataset finished")

    collator_fn = DefaultCollator().generative_r2d2_collate_fn_ext
    from torch.utils.data import DataLoader, SequentialSampler
    dataloader = DataLoader(bllip_dataset, batch_size=2, sampler=SequentialSampler(bllip_dataset),
                            collate_fn=collator_fn, num_workers=0)
    
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, inputs in enumerate(epoch_iterator):
        print("step: ", step)
        # print(inputs)
        # print(len(inputs['chunk_parses'][0]), len(inputs['chunk_parses'][0]))