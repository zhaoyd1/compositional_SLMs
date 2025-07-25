import os
import numpy as np
import pickle
import nltk
import codecs
import tarfile
from itertools import accumulate
import re
import argparse
import sys


def build_dataset(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line]
                for sent in sents:
                    ids = tokenizer.encode(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_dataset_tg(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None, closing_start_idx=99999):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    transformed_index_path = os.path.join(output_dir, f'data.transformed_len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    transformed_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            extra_len = sum([1 if idx >= closing_start_idx else 0 for idx in ids])
            transformed_lens.append(len(ids) + extra_len)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    transformed_lens.append(-1)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line]
                for sent in sents:
                    ids = tokenizer.encode(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)
    
    with open(transformed_index_path, mode='wb') as index_out:
        pickle.dump(transformed_lens, index_out)

def build_dataset_gpst(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None, closing_start_idx=99999):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    transformed_index_path = os.path.join(output_dir, f'data.transformed_len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    transformed_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            extra_len = sum([1 if idx >= closing_start_idx else 0 for idx in ids])
            transformed_lens.append(len(ids) + extra_len)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    transformed_lens.append(-1)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line.split()]
                for sent in sents:
                    ids = tokenizer.convert_tokens_to_ids(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)
    
    with open(transformed_index_path, mode='wb') as index_out:
        pickle.dump(transformed_lens, index_out)

def build_dataset_0010(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None, opening_start_idx=99999, closing_start_idx=99999):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    transformed_index_path = os.path.join(output_dir, f'data.transformed_len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    transformed_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            op_nt_num = sum([1 if (idx < closing_start_idx and idx >= opening_start_idx) else 0 for idx in ids])
            transformed_lens.append(len(ids) - op_nt_num)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    transformed_lens.append(-1)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line.split()]
                for sent in sents:
                    ids = tokenizer.convert_tokens_to_ids(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)
    
    with open(transformed_index_path, mode='wb') as index_out:
        pickle.dump(transformed_lens, index_out)

def build_dataset_1010(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None, opening_start_idx=99999, closing_start_idx=99999):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    transformed_index_path = os.path.join(output_dir, f'data.transformed_len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    transformed_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            # op_nt_num = sum([1 if (idx < closing_start_idx and idx >= opening_start_idx) else 0 for idx in ids])
            # transformed_lens.append(len(ids) - op_nt_num)
            transformed_lens.append(len(ids))
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in f_in:
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
                    item_lens.append(0)
                    transformed_lens.append(-1)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                    else:
                        sents = sent_tokenizer(line)
                else:
                    sents = [line.split()]
                for sent in sents:
                    ids = tokenizer.convert_tokens_to_ids(sent)
                    current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)
    
    with open(transformed_index_path, mode='wb') as index_out:
        pickle.dump(transformed_lens, index_out)

def build_dataset_gpst_0001(text_path, text_path_raw, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False, sent_tokenizer=None, closing_start_idx=99999):
    filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    seq_index_path = os.path.join(output_dir, f'data.seq.len.pkl')
    content_path = os.path.join(output_dir, f'data_seq')
    content_path_raw = os.path.join(output_dir, f'data')
    item_lens = []
    seq_item_lens = []
    current_size = buffer_size
    current_offset = 0
    current_size_raw = buffer_size
    current_offset_raw = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    np_memmap_raw = np.memmap(content_path_raw, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    current_buffer_raw = []

    def flush(buffer, buffer_raw):
        nonlocal np_memmap
        nonlocal np_memmap_raw
        nonlocal current_size
        nonlocal current_offset
        nonlocal current_offset_raw
        nonlocal current_size_raw

        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        
        total_len = 0
        for ids in current_buffer_raw:
            total_len += len(ids)
        while current_offset_raw + total_len > current_size_raw:
            # expand
            np_memmap_raw.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size_raw
            np_memmap_raw = np.memmap(content_path_raw, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size_raw = next_size
        
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            seq_item_lens.append(len(ids))
        
        for ids in current_buffer_raw:
            np_memmap_raw[current_offset_raw: current_offset_raw + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset_raw += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset, current_offset_raw
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in, open(text_path_raw, mode='r') as f_in_raw:
        for line, line_raw in zip(f_in, f_in_raw):
            line = line.strip()
            line_raw = line_raw.strip()
            if len(line) == 0: # document split
                assert len(line_raw) == 0
                if len(current_buffer) > 0:
                    flush(current_buffer, current_buffer_raw)
                    current_buffer = []
                    current_buffer_raw = []
                    item_lens.append(0)
                    seq_item_lens.append(0)
                    # doc_num += 1
                    # if doc_num > 10:
                    #     break
            else:
                # tokenize to ids
                if tokenize_sent:
                    if sent_tokenizer is None:
                        sents = nltk.sent_tokenize(line)
                        sents_raw = nltk.sent_tokenize(line_raw)
                    else:
                        sents = sent_tokenizer(line)
                        sents_raw = sent_tokenizer(line_raw)
                else:
                    sents = [line.split()]
                    sents_raw = [line_raw]
                for sent in sents:
                    ids = tokenizer.convert_tokens_to_ids(sent)
                    current_buffer.append(ids)
                for sent in sents_raw:
                    ids = tokenizer.encode(sent)
                    current_buffer_raw.append(ids)
                assert len(current_buffer[-1]) == 2 * len(current_buffer_raw[-1]) - 1
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
                # current_buffer.append(ids)
            
        if len(current_buffer) > 0:
            flush(current_buffer, current_buffer_raw)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)
    
    with open(seq_index_path, mode='wb') as index_out:
        pickle.dump(seq_item_lens, index_out)

def build_dataset_from_dir(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1, tokenize_sent=False):
    # filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    # with open(text_path, mode='r') as f_in:
    processed_files = 0
    for root, dirs, files in os.walk(texts_dir):
        for text_path in files:
            if processed_files % 10 == 0:
                print(f'processed: {processed_files} / {len(files)}', flush=True)
            processed_files += 1
            if text_path.endswith('_data'):
                tar = tarfile.open(os.path.join(root, text_path))
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    lines = file.readlines()
                    for line in lines:
                        line = line.decode().strip()
                        if len(line) == 0: # document split
                            if len(current_buffer) > 0:
                                flush(current_buffer)
                                current_buffer = []
                                item_lens.append(0)
                                # doc_num += 1
                                # if doc_num > 10:
                                #     break
                        else:
                            # tokenize to ids
                            if tokenize_sent:
                                sents = nltk.sent_tokenize(line)
                            else:
                                sents = [line]
                            for sent in sents:
                                ids = tokenizer.encode(sent)
                                current_buffer.append(ids)
                            if max_len > 0 and len(ids) >= max_len:
                                #drop
                                continue
                        
                    if len(current_buffer) > 0:
                        flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_dataset_batch(text_path_pattern, tokenizer, output_dir, buffer_size = 1024, max_len=200):
    import glob
    files = glob.glob(text_path_pattern)
    for f in files:
        pass

def print_dataset(index_path, data_path, tokenizer):
    np_memmap = np.memmap(data_path, dtype=np.int32, mode='r', order='C')
    with open(index_path, 'rb') as handle:
        item_lens = pickle.load(handle)
    
    ends = list(accumulate(item_lens))
    prev_end = 0
    for end in ends:
        print(tokenizer.convert_ids_to_tokens(np_memmap[prev_end : end]))
        prev_end = end


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocess corpus components')
    cmd.add_argument('--mode', required=True, choices=['wikitext103', 'openwebtext', 'bllip', 'bllip_tg', 'bllip_gpst', 'bllip_0001', 'bllip_0010', 'bllip_1010'], default='wikitext103')
    cmd.add_argument('--tokenizer_config_path', required=True, type=str, help='config for tokenizer')
    cmd.add_argument('--raw_corpus_path', required=True, type=str, help='path for raw corpus')
    cmd.add_argument('--output_path', required=True, type=str, help='path for preprocessed corpus, end with .lazy')
    cmd.add_argument('--special_label', required=True, type=int, default=0, help='special label for non-terminal')
    cmd.add_argument('--data_raw_path', required=False, type=str, help='path for raw corpus')
    args = cmd.parse_args(sys.argv[1:])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config_path)
    from tokenizers import AddedToken
    if args.special_label != 0:
        non_terminal_token = ['(ADJP', '(ADVP', '(CONJP', '(FRAG', '(INTJ', '(LST', '(NAC', '(NP', '(NX', '(PP', '(PRN', '(PRT', '(QP', '(RRC', '(S', '(SBAR', '(SBARQ', '(SINV', '(SQ', '(UCP', '(VP', '(WHADJP', '(WHADVP', '(WHNP', '(WHPP', '(X', \
                              'ADJP)', 'ADVP)', 'CONJP)', 'FRAG)', 'INTJ)', 'LST)', 'NAC)', 'NP)', 'NX)', 'PP)', 'PRN)', 'PRT)', 'QP)', 'RRC)', 'S)', 'SBAR)', 'SBARQ)', 'SINV)', 'SQ)', 'UCP)', 'VP)', 'WHADJP)', 'WHADVP)', 'WHNP)', 'WHPP)', 'X)']
        tokens = [AddedToken(token, lstrip=True) for token in non_terminal_token]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        closing_start_idx = tokenizer.get_vocab()["ADJP)"]
        opening_start_idx = tokenizer.get_vocab()["(ADJP"]
    if args.mode == "wikitext103":
        build_dataset(args.raw_corpus_path, tokenizer, args.output_path, 
                buffer_size=16384, max_len=-1, tokenize_sent=True)
    elif args.mode == "openwebtext":
        build_dataset_from_dir(args.raw_corpus_path, tokenizer, args.output_path, 
                 buffer_size=16384, max_len=-1, tokenize_sent=True)
    elif args.mode == "bllip":
        build_dataset(args.raw_corpus_path, tokenizer, args.output_path, 
                buffer_size=16384, max_len=-1, tokenize_sent=False)
    elif args.mode == "bllip_tg":
        build_dataset_tg(args.raw_corpus_path, tokenizer, args.output_path, 
                buffer_size=16384, max_len=-1, tokenize_sent=False, closing_start_idx=closing_start_idx)
    elif args.mode == "bllip_gpst":
        build_dataset_gpst(args.raw_corpus_path, tokenizer, args.output_path,
                buffer_size=16384, max_len=-1, tokenize_sent=False, closing_start_idx=closing_start_idx)
    elif args.mode == "bllip_0001":
        build_dataset_gpst_0001(args.raw_corpus_path, args.data_raw_path, tokenizer, args.output_path,
                buffer_size=16384, max_len=-1, tokenize_sent=False, closing_start_idx=closing_start_idx)
    elif args.mode == "bllip_0010":
        build_dataset_0010(args.raw_corpus_path, tokenizer, args.output_path,
                buffer_size=16384, max_len=-1, tokenize_sent=False, opening_start_idx=opening_start_idx, closing_start_idx=closing_start_idx)
    elif args.mode == "bllip_1010":
        build_dataset_1010(args.raw_corpus_path, tokenizer, args.output_path,
                buffer_size=16384, max_len=-1, tokenize_sent=False, opening_start_idx=opening_start_idx, closing_start_idx=closing_start_idx)
    else:
        raise Exception('Mode not suppport')