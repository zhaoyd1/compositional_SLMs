# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
import os
import logging
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from model.model_factory import create_model
from reader.lazy_loader import LazyLoader
from reader.dataset_gpt_tg_seq import GPT2Dataset
from torch.utils.data.distributed import DistributedSampler
from reader.data_collator import DefaultCollator
from utils.model_loader import get_max_epoch_step, load_checkpoint
from utils.tree_utils import get_tree_from_merge_trajectory
from utils.misc import gpt_token
from model.weighted_sum_func import WeightedSumFunc
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoModel, AutoConfig
from masking_gpst import utils as masking_utils
from masking_gpst import masking_types as types
from tokenizers import AddedToken
from typing import Optional

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _scalar(val):
    if val is not None:
        if isinstance(val, torch.Tensor):
            return val.item()
        return val
    return 0

class LinearProgressScheduler:
    def __init__(self, start, end, proportion, total_steps):
        # e.g. proportion = 0.8
        # then val will go from start to end at previous 80% steps and keep end in the last 20% steps
        self._start = start
        self._end = end
        self._total_steps = total_steps * proportion

    def update(self, current_step):
        r = min(1.0, current_step / self._total_steps)
        return self._start * (1 - r) + self._end * r


class Trainer(object):
    def __init__(self, 
                 model,
                 collator,
                 tokenizer,
                 device,
                 logger,
                 is_master=True,
                 num_workers = 0,
                 lr=5e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger
        self.collator = collator
        self.num_workers = num_workers

        self.device = device
        self.lr = lr

    def train(self, 
              data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              output_dir,
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              log_steps=100, save_steps=100, 
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1,
              eval_steps=1000,
              dev_path='corpus/bllip',
              left=0):

        total_step = len(data_loader)

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.train()

        for step, inputs in enumerate(epoch_iterator):
            if step <= max_recover_step:
                continue
            max_recover_step = -1

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            # for key, val in inputs.items():
            #     logger.info(f'{key}: {val}')
            # exit()
            coeff = 1.0 if coeff_scheduler is None else coeff_scheduler.update(step)
            temperature = 1.0 if temp_scheduler is None else temp_scheduler.update(step)
            with model.no_sync():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    result = self.model(**inputs, coeff=coeff, temperature=temperature)
                
                # scaler.scale((result.struct_loss + result.non_struct_loss) / accumulation_steps).backward()
                if result.struct_loss is not None:
                    WeightedSumFunc.a_ij_require_grad = True
                    scaler.scale(result.struct_loss / accumulation_steps).backward(retain_graph=True)
                # if (step + 1) % accumulation_steps != 0:
                WeightedSumFunc.a_ij_require_grad = False
                scaler.scale(result.non_struct_loss / accumulation_steps).backward()

            # try:
            if (step + 1) % accumulation_steps == 0:
                # for p in model.parameters():
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= torch.distributed.get_world_size()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                    # print(f'is master: {self.is_master}, param0: {next(model.parameters())[:5]}')
            # except RuntimeError as e:
            #     self.logger.error(e)
            # finally:
            if (step + 1) % accumulation_steps == 0:
                optimizer.zero_grad()

            if step % log_steps == 0 and step > 0:
                self.logger.info(f'progress:{step}/{total_step} coeff: {coeff} loss: {_scalar(result.non_struct_loss + result.struct_loss)} gpt loss: {_scalar(result.gpt_loss)} ' + \
                    f'inside_outside loss: {_scalar(result.inside_outside_loss)} parser loss: {_scalar(result.parser_loss)} ' + \
                    f'action loss: {_scalar(result.action_loss)}')
                # with torch.no_grad():
                #     # output generated binary tree
                #     if self.is_master and result.splits is not None:
                #         # output binary trees for different iteration epochs
                #         # sent_id = np.random.randint(inputs['input_ids'].shape[0])
                #         sent_id = 0
                #         seq_len = inputs["masks"][sent_id].sum()
                #         input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][sent_id].cpu().data.numpy())
                #         self.logger.info(f"input sentence: {input_tokens}")
                #         tokens = [gpt_token(t) for t in input_tokens]
                #         print(seq_len, sent_id, result.splits)
                #         split_points = [_ for _ in reversed(result.splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                #         merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                #         self.logger.info(f"parsed tree : {merged_tree}")
            if step % eval_steps == 0:
                if left == 1:
                    self.evaluate_dev(os.path.join(dev_path, 'bllip_dev.txt'), os.path.join(dev_path, 'dev_left_parses.txt'))
                else:
                    self.evaluate_dev(os.path.join(dev_path, 'bllip_dev.txt'), os.path.join(dev_path, 'dev_new_parses.txt'))

            if step % save_steps == 0 and step > 0:
                try:
                    torch.save(self.model.state_dict(),
                            os.path.join(output_dir, f"model0_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer0_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler0_{step}.pt"))
                    
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler0_{step}.pt'))
                except:
                    pass

        if self.is_master:
            while True:
                try:
                    torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))
                    break
                except:
                    time.sleep(5)

    def train_tg(self, 
              data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              output_dir,
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              log_steps=100, save_steps=100, 
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1,
              eval_steps=1000,
              dev_path='corpus/bllip',
              ranges: Optional[masking_utils.TokenTypeRanges] = None,
              mask_rules = None):

        total_step = len(data_loader)

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.train()

        for step, inputs in enumerate(epoch_iterator):
            if step <= max_recover_step:
                continue
            max_recover_step = -1
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            # for key, val in inputs.items():
            #     logger.info(f'{key}: {val}')
            # exit()
            coeff = 1.0 if coeff_scheduler is None else coeff_scheduler.update(step)
            temperature = 1.0 if temp_scheduler is None else temp_scheduler.update(step)
            with model.no_sync():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    result = self.model(**inputs, coeff=coeff, temperature=temperature)
                
                # scaler.scale((result.struct_loss + result.non_struct_loss) / accumulation_steps).backward()
                # if result.struct_loss is not None:
                #     WeightedSumFunc.a_ij_require_grad = True
                #     scaler.scale(result.struct_loss / accumulation_steps).backward(retain_graph=True)
                # if (step + 1) % accumulation_steps != 0:
                # WeightedSumFunc.a_ij_require_grad = False
                scaler.scale(result.non_struct_loss / accumulation_steps).backward()

            # try:
            if (step + 1) % accumulation_steps == 0:
                # for p in model.parameters():
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= torch.distributed.get_world_size()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                    # print(f'is master: {self.is_master}, param0: {next(model.parameters())[:5]}')
            # except RuntimeError as e:
            #     self.logger.error(e)
            # finally:
            if (step + 1) % accumulation_steps == 0:
                optimizer.zero_grad()

            if step % log_steps == 0 and step > 0:
                self.logger.info(f'progress:{step}/{total_step} coeff: {coeff} loss: {_scalar(result.non_struct_loss)}')
                # with torch.no_grad():
                #     # output generated binary tree
                #     if self.is_master and result.splits is not None:
                #         # output binary trees for different iteration epochs
                #         # sent_id = np.random.randint(inputs['input_ids'].shape[0])
                #         sent_id = 0
                #         seq_len = inputs["masks"][sent_id].sum()
                #         input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][sent_id].cpu().data.numpy())
                #         self.logger.info(f"input sentence: {input_tokens}")
                #         tokens = [gpt_token(t) for t in input_tokens]
                #         print(seq_len, sent_id, result.splits)
                #         split_points = [_ for _ in reversed(result.splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                #         merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                #         self.logger.info(f"parsed tree : {merged_tree}")
            # if step % eval_steps == 0:
            #     self.evaluate_tg_dev(os.path.join(dev_path, 'dev_new.cc'), ranges, mask_rules)

            if step % save_steps == 0 and step > 0:
                try:
                    torch.save(self.model.state_dict(),
                            os.path.join(output_dir, f"model0_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer0_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler0_{step}.pt"))
                    
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler0_{step}.pt'))
                except:
                    pass

        if self.is_master:
            while True:
                try:
                    torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))
                    break
                except:
                    time.sleep(5)
    
    def train_sent(self, 
              text_data,
              parses, 
              optimizer, 
              scheduler, 
              scaler,
              output_dir,
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              epoch=10,
              batch_size=8,
              log_steps=100, save_steps=100, 
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1,
              eval_steps=1000,
              dev_path='corpus/bllip',
              left=False):

        # with open(train_text_data, 'r') as f:
        #     text_data = f.readlines()
        # text_data = [t.strip() for t in text_data]
        # # with ProcessPoolExecutor(max_workers=8) as executor:
        # #     text_data = list(executor.map(self.tokenizer.encode, tqdm(text_data, desc='tokenize')))
        # text_data = [self.tokenizer.encode(t) for t in tqdm(text_data, desc='tokenize')]
        # parses = []
        # with open(train_parse_data, 'r') as f:
        #     parses = f.readlines()
        #     parses = [eval(p.strip()) for p in parses]
        # assert len(text_data) == len(parses)
        # for i in range(len(text_data)):
        #     if len(text_data[i]) != len(parses[i]) + 1:
        #         logger.info(f'error: {len(text_data[i])} != {len(parses[i]) + 1}')
        #         logger.info(f'{i}')
        # exit()
        total_step = (epoch * len(text_data)) // batch_size  
        
        self.model.train()
        for step in tqdm(range(total_step)):
            
            inputs = {"chunk_input_ids": None, "chunk_masks": None, "input_ids": None,
                        "masks": None, "group_ids": None, "chunk_parses": None}
            id_start = step * batch_size
            id_end = (step + 1) * batch_size
            ids = [i % len(text_data) for i in range(id_start, id_end)]
            chunk_num = len(ids)
            max_sent_len = max([len(text_data[i]) for i in ids])
            chunk_input_ids = [text_data[i] + [-100] * (max_sent_len - len(text_data[i])) for i in ids]
            chunk_input_ids = torch.tensor(chunk_input_ids, dtype=torch.long).to(self.device)
            chunk_masks = [[1] * len(text_data[i]) + [0] * (max_sent_len - len(text_data[i])) for i in ids]
            chunk_masks = torch.tensor(chunk_masks, dtype=torch.long).to(self.device)
            input_ids = [text_data[i] + [0] * (max_sent_len - len(text_data[i])) for i in ids]
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            masks = [[1] * len(text_data[i]) + [0] * (max_sent_len - len(text_data[i])) for i in ids]
            masks = torch.tensor(masks, dtype=torch.long).to(self.device)
            group_ids = np.arange(chunk_num)
            chunk_parses = [parses[i] + [p for p in range(max(len(text_data[i])-1, 0), max_sent_len - 1)] for i in ids]
            chunk_parses = torch.tensor(chunk_parses, dtype=torch.long).to(self.device)
            inputs['chunk_input_ids'] = chunk_input_ids
            inputs['chunk_masks'] = chunk_masks
            inputs['input_ids'] = input_ids
            inputs['masks'] = masks
            inputs['group_ids'] = group_ids
            inputs['chunk_parses'] = chunk_parses

            # for key, val in inputs.items():
            #     logger.info(f'{key}: {val}')
            # exit()
            coeff = 1.0 if coeff_scheduler is None else coeff_scheduler.update(step)
            temperature = 1.0 if temp_scheduler is None else temp_scheduler.update(step)
            with model.no_sync():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    result = self.model(**inputs, coeff=coeff, temperature=temperature)
                
                # scaler.scale((result.struct_loss + result.non_struct_loss) / accumulation_steps).backward()
                if result.struct_loss is not None:
                    WeightedSumFunc.a_ij_require_grad = True
                    scaler.scale(result.struct_loss / accumulation_steps).backward(retain_graph=True)
                # if (step + 1) % accumulation_steps != 0:
                WeightedSumFunc.a_ij_require_grad = False
                scaler.scale(result.non_struct_loss / accumulation_steps).backward()

            # try:
            if (step + 1) % accumulation_steps == 0:
                # for p in model.parameters():
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= torch.distributed.get_world_size()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                    # print(f'is master: {self.is_master}, param0: {next(model.parameters())[:5]}')
            # except RuntimeError as e:
            #     self.logger.error(e)
            # finally:
            if (step + 1) % accumulation_steps == 0:
                optimizer.zero_grad()

            if step % log_steps == 0 and step > 0:
                self.logger.info(f'progress:{step}/{total_step} coeff: {coeff} loss: {_scalar(result.non_struct_loss + result.struct_loss)} gpt loss: {_scalar(result.gpt_loss)} ' + \
                    f'inside_outside loss: {_scalar(result.inside_outside_loss)} parser loss: {_scalar(result.parser_loss)} ' + \
                    f'action loss: {_scalar(result.action_loss)}')
                # with torch.no_grad():
                #     # output generated binary tree
                #     if self.is_master and result.splits is not None:
                #         # output binary trees for different iteration epochs
                #         # sent_id = np.random.randint(inputs['input_ids'].shape[0])
                #         sent_id = 0
                #         seq_len = inputs["masks"][sent_id].sum()
                #         input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][sent_id].cpu().data.numpy())
                #         self.logger.info(f"input sentence: {input_tokens}")
                #         tokens = [gpt_token(t) for t in input_tokens]
                #         print(seq_len, sent_id, result.splits)
                #         split_points = [_ for _ in reversed(result.splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                #         merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                #         self.logger.info(f"parsed tree : {merged_tree}")
            if step % eval_steps == 0:
                if left == 1:
                    self.evaluate_dev(os.path.join(dev_path, 'bllip_dev.txt'), os.path.join(dev_path, 'supar_dev_parses.txt'))
                else:
                    self.evaluate_dev(os.path.join(dev_path, 'bllip_dev.txt'), os.path.join(dev_path, 'dev_new_parses.txt'))

            if step % save_steps == 0 and step > 0:
                try:
                    torch.save(self.model.state_dict(),
                            os.path.join(output_dir, f"model0_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer0_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler0_{step}.pt"))
                    
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler0_{step}.pt'))
                except:
                    pass

        if self.is_master:
            while True:
                try:
                    torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))
                    break
                except:
                    time.sleep(5)
    
    # def evaluate_tg_dev(self, text_data, ranges: Optional[masking_utils.TokenTypeRanges] = None,
    #           mask_rules = None):
    #     with open(text_data, 'r') as f:
    #         text_data = f.readlines()
    #     text_data = [t.strip() for t in text_data]
    #     text_data = [self.tokenizer.encode(t) for t in tqdm(text_data, desc='tokenize')]
    #     closing_nt = ranges.closing_non_terminals
    #     bos_id = ranges.start_token
    #     self.model.eval()
    #     total_loss = 0
    #     total_length = 0

    #     with torch.no_grad():
    #         for i in range(0, len(text_data), 30):
    #             cur_trans_len = []
    #             input_ids = []
    #             tgt_ids = []
    #             attn_masks = []
    #             rel_positions = []
    #             for j in range(i, min(i + 30, len(text_data))):
    #                 extra_length = sum([1 for t in text_data[j] if (t >= closing_nt[0]) and (t < closing_nt[1])])
    #                 transformed_length = len(text_data[j]) + extra_length
    #                 cur_trans_len.append(transformed_length)
    #                 total_length += len(text_data[j])
    #             max_len = max(cur_trans_len)
                
    #             for j in range(i, min(i + 30, len(text_data))):
    #                 st = time.time()
    #                 src_ = torch.LongTensor([bos_id] + text_data[j][:-1])
    #                 tgt_ = torch.LongTensor(text_data[j])
    #                 info_tuple = masking_utils.compute_token_types(
    #                     {"inputs": src_, "labels": tgt_}, ranges
    #                 )
    #                 st_1 = time.time()
    #                 logger.info(f'compute token types time: {st_1 - st}')
    #                 chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
    #                                                         info_tuple['labels'], info_tuple['labels_ttypes'])
    #                 st_2 = time.time()
    #                 logger.info(f'chunks for sequence time: {st_2 - st_1}')
    #                 chunks = [types.Chunk(None, *chunk) for chunk in chunks]
    #                 chunk = chunks[0]
    #                 src_p = chunk.inputs[:max_len]
    #                 tgt_p = chunk.labels[:max_len]
    #                 mask = chunk.attn_mask[:max_len, :max_len]
    #                 for i in range(len(mask)):
    #                     mask[i, i] = 1
    #                 chunk_len = len(chunk.inputs)
    #                 relpos = chunk.attn_relpos[:max_len, chunk_len:chunk_len+max_len]
    #                 input_ids.append(np.array(src_p))
    #                 tgt_ids.append(np.array(tgt_p))
    #                 attn_masks.append(np.array(mask))
    #                 rel_positions.append(np.array(relpos))
    #                 st_3 = time.time()
    #                 logger.info(f'append time: {st_3 - st_2}')
                
    #             inputs = {"chunk_input_ids": torch.tensor(np.array(input_ids, dtype=np.int32), dtype=torch.long),
    #                       "chunk_tgt_ids": torch.tensor(np.array(tgt_ids, dtype=np.int32), dtype=torch.long),
    #                       "attn_masks": torch.tensor(np.array(attn_masks, dtype=np.int32), dtype=torch.bool),
    #                       "rel_positions": torch.tensor(np.array(rel_positions, dtype=np.int32), dtype=torch.long)}
    #             st_4 = time.time()
    #             logger.info(f'prepare inputs time: {st_4 - st_3}')
    #             for key, val in inputs.items():
    #                 inputs[key] = val.to(self.device)
    #             # inputs["attn_masks"] = None
    #             logger.info("111")
  
    #             result = self.model(**inputs, tg_inference=True)
    #             loss = result.non_struct_loss
    #             loss = loss.sum().sum().item()
    #             total_loss += loss
    #             logger.info("222")
        
    #     self.logger.info(f'dev loss: {total_loss / total_length}')
    #     self.model.train()   

    def evaluate_dev(self, text_data, parse_data):
        with open(text_data, 'r') as f:
            text_data = f.readlines()
        text_data = [t.strip() for t in text_data]
        text_data = [self.tokenizer.encode(t) for t in tqdm(text_data, desc='tokenize')]
        parses = []
        with open(parse_data, 'r') as f:
            parses = f.readlines()
            parses = [eval(p.strip()) for p in parses]
        assert len(text_data) == len(parses)
        self.model.eval()
        total_gpt_loss = []
        total_action_loss = []
        total_gpt_len = 0
        total_action_len = 0
        with torch.no_grad():
            for i in range(0, len(text_data), 30):
                for j in range(i, min(i + 30, len(text_data))):
                    assert len(text_data[j]) == len(parses[j]) + 1

                inputs = {"chunk_input_ids": None, "chunk_masks": None, "input_ids": None,
                            "masks": None, "group_ids": None, "chunk_parses": None}
                chunk_num = 30
                max_sent_len = max([len(t) for t in text_data[i:i+chunk_num]])
                total_gpt_len += sum([len(t) for t in text_data[i:i+chunk_num]])
                total_action_len += 2 * sum([len(p) for p in parses[i:i+chunk_num]])
                chunk_input_ids = [text_data[j] + [-100] * (max_sent_len - len(text_data[j])) for j in range(i, i + chunk_num)]
                chunk_input_ids = torch.tensor(chunk_input_ids, dtype=torch.long).to(self.device)
                chunk_masks = [[1] * len(text_data[j]) + [0] * (max_sent_len - len(text_data[j])) for j in range(i, i + chunk_num)]
                chunk_masks = torch.tensor(chunk_masks, dtype=torch.long).to(self.device)
                input_ids = [text_data[j] + [0] * (max_sent_len - len(text_data[j])) for j in range(i, i + chunk_num)]
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
                masks = [[1] * len(text_data[j]) + [0] * (max_sent_len - len(text_data[j])) for j in range(i, i + chunk_num)]
                masks = torch.tensor(masks, dtype=torch.long).to(self.device)
                group_ids = np.arange(chunk_num)
                chunk_parses = [parses[j] + [p for p in range(max(len(text_data[j])-1, 0), max_sent_len - 1)] for j in range(i, i + chunk_num)]
                chunk_parses = torch.tensor(chunk_parses, dtype=torch.long).to(self.device)
                inputs['chunk_input_ids'] = chunk_input_ids
                inputs['chunk_masks'] = chunk_masks
                inputs['input_ids'] = input_ids
                inputs['masks'] = masks
                inputs['group_ids'] = group_ids
                inputs['chunk_parses'] = chunk_parses
                result = self.model(**inputs, ppl=True)
                gpt_loss = result.gpt_loss.sum().sum().item()
                action_loss = result.action_loss.sum().sum().item()
                total_gpt_loss.append(gpt_loss)
                total_action_loss.append(action_loss)
        
        self.logger.info(f'dev gpt loss: {sum(total_gpt_loss) / total_gpt_len}')
        self.logger.info(f'dev action loss: {sum(total_action_loss) / total_action_len}')
        self.model.train()
    
    def evaluate_doc(self, text_data, tg_seq_data, ranges, mask_rules):
        with open(text_data, 'r') as f:
            text_data = f.readlines()
        
        all_tg_seq = []
        cur_seq = []
        cur_seq_set = set()
        with open(tg_seq_data, 'r') as f:
            tg_seq_data = f.readlines()
            for i in  tqdm(range(len(tg_seq_data))):
                line = tg_seq_data[i].strip()
                if line.startswith('Sentence'):
                    if len(cur_seq) > 0:
                        all_tg_seq.append(cur_seq)
                        cur_seq = []
                        cur_seq_set = set()
                    continue
                if len(cur_seq) >= 300:
                    continue
                
                if line.strip() not in cur_seq_set:
                    cur_seq_set.add(line.strip())
                    cur_line = line.strip().split()
                    seq = self.tokenizer.convert_tokens_to_ids(cur_line)
                    cur_seq.append(seq)

            if len(cur_seq) > 0:
                all_tg_seq.append(cur_seq)
        
        cur_prefix = []
        cur_prefix_loss = 0
        total_loss = 0
        cur_doc_loss = 0
        total_token_length = 0
        cnt = 0
        self.model.eval()
        doc_num = 0
        prev_doc_num = -1
        prev_seq_len = 0
        fail_num = 0
        exclude = False
        with torch.no_grad():
            for i in tqdm(range(len(text_data))):
                if text_data[i].strip() == "":
                    logger.info("new doc")
                    cur_prefix = []
                    cur_prefix_loss = 0
                    total_loss += cur_doc_loss
                    cur_doc_loss = 0
                    doc_num += 1
                    prev_seq_len = 0
                    exclude = False
                    continue
                if exclude:
                    cnt += 1
                    continue

                # cur_text = self.tokenizer.encode(text_data[i].strip())
                # cur_sent_len = len(cur_text)
                
                cur_seq = all_tg_seq[cnt]
                cur_sent_len = sum([1 for t in cur_seq[0] if t < ranges.start_token])
                # if len(cur_text) != len(cur_parses[0]) + 1:
                #     logger.info(f'error: {len(cur_text)} != {len(cur_parses[0]) + 1}')
                #     logger.info(f'{cur_text}')
                #     logger.info(f'{cur_parses[0]}')
                #     exit()
                # if len(cur_text) * 2 - 1 != len(cur_seq[0]):
                #     logger.info(f'error: {len(cur_text) * 2 - 1} != {len(cur_seq[0])}')
                #     logger.info(f'{cur_text}')
                #     logger.info(f'{cur_seq[0]}')
                #     exit()  
                
                if cur_prefix != []:
                    cur_seq =[cur_prefix + seq for seq in cur_seq]
                seq_len = max([len(seq) for seq in cur_seq])

                if seq_len > 2047:
                    logger.info(f'exceed 2048 error: {seq_len}, {cnt}')
                    cnt += 1
                    if doc_num != prev_doc_num:
                        fail_num += 1
                        prev_doc_num = doc_num
                    exclude = True
                    continue
                
                seq_num = len(cur_seq)
                total_token_length += cur_sent_len
                cur_loss = None
                
                if (seq_len > 150 and seq_num > 20) or (seq_len > 100 and seq_num > 50):
                    for j in range(0, seq_num, 25):
                        chunk_input_ids = []
                        chunk_tgt_ids = []
                        chunk_masks = []
                        rel_positions = [] 
                        tmp_cur_seq = cur_seq[j:j+25]
                        inputs = {"chunk_input_ids": None, "chunk_tgt_ids": None}
                        
                        with ProcessPoolExecutor(max_workers=8) as executor:
                            results = executor.map(process_seq, tmp_cur_seq, [seq_len] * len(tmp_cur_seq))

                        for result in results:
                            src_p, tgt_p = result
                            chunk_input_ids.append(src_p)
                            chunk_tgt_ids.append(tgt_p)
                        # for seq in tmp_cur_seq:   
                        #     src_ = torch.LongTensor([ranges.start_token] + seq[:-1])
                        #     tgt_ = torch.LongTensor(seq)
                        #     info_tuple = masking_utils.compute_token_types(
                        #         {"inputs": src_, "labels": tgt_}, ranges
                        #     )
                        #     chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                        #                                             info_tuple['labels'], info_tuple['labels_ttypes'])

                        #     chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                        #     chunk = chunks[0]
                        #     src_p = chunk.inputs[:seq_len]
                        #     tgt_p = chunk.labels[:seq_len]
                        #     mask = chunk.attn_mask[:seq_len, :seq_len]
                        #     for i in range(len(mask)):
                        #         mask[i, i] = 1
                        #     chunk_len = len(chunk.inputs)
                        #     relpos = chunk.attn_relpos[:seq_len, chunk_len:chunk_len+seq_len]
                        #     chunk_input_ids.append(src_p)
                        #     chunk_tgt_ids.append(tgt_p)
                        #     chunk_masks.append(mask)
                        #     rel_positions.append(relpos)
                        
                        inputs['chunk_input_ids'] = torch.tensor(np.array(chunk_input_ids), dtype=torch.long).to(self.device)
                        inputs['chunk_tgt_ids'] = torch.tensor(np.array(chunk_tgt_ids), dtype=torch.long).to(self.device)
                        
                        with model.no_sync():
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                result = self.model(**inputs, tg_inference=True)
                        if cur_loss is None:
                            cur_loss = result.non_struct_loss.detach().cpu()
                        else:
                            cur_loss = torch.cat([cur_loss, result.non_struct_loss.detach().cpu()], dim=0)
                else:
                    inputs = {"chunk_input_ids": None, "chunk_tgt_ids": None}
                    chunk_input_ids = []
                    chunk_tgt_ids = []
                    chunk_masks = []
                    rel_positions = []
                    with ProcessPoolExecutor(max_workers=8) as executor:
                        results = executor.map(process_seq, cur_seq, [seq_len] * len(cur_seq))
                    
                    for result in results:
                        src_p, tgt_p = result
                        chunk_input_ids.append(src_p)
                        chunk_tgt_ids.append(tgt_p)
                    # for seq in cur_seq:
                    #     src = torch.LongTensor([ranges.start_token] + seq[:-1])
                    #     tgt = torch.LongTensor(seq)
                    #     info_tuple = masking_utils.compute_token_types(
                    #         {"inputs": src, "labels": tgt}, ranges
                    #     )
                    #     chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                    #                                             info_tuple['labels'], info_tuple['labels_ttypes'])
                    #     chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                    #     chunk = chunks[0]
                    #     src_p = chunk.inputs[:seq_len]
                    #     tgt_p = chunk.labels[:seq_len]
                    #     mask = chunk.attn_mask[:seq_len, :seq_len]
                    #     for k in range(len(mask)):
                    #         mask[k, k] = 1
                    #     chunk_len = len(chunk.inputs)
                    #     relpos = chunk.attn_relpos[:seq_len, chunk_len:chunk_len+seq_len]
                    #     chunk_input_ids.append(src_p)
                    #     chunk_tgt_ids.append(tgt_p)
                    #     chunk_masks.append(mask)
                    #     rel_positions.append(relpos)
                    
                    inputs['chunk_input_ids'] = torch.tensor(np.array(chunk_input_ids), dtype=torch.long).to(self.device)
                    inputs['chunk_tgt_ids'] = torch.tensor(np.array(chunk_tgt_ids), dtype=torch.long).to(self.device)

                    with model.no_sync():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            result = self.model(**inputs, tg_inference=True)
                    # logger.info(result.action_loss)
                    # logger.info(result.gpt_loss)
                    # logger.info(cur_text)
                    # logger.info(len(cur_parses))
                    # if cur_prefix_loss != 0:
                    #     logger.info(cur_prefix_loss)
                    #     logger.info(len(cur_text) - len(cur_prefix))
                    #     prefix_gpt_loss = result.gpt_loss[:, :len(cur_prefix)].sum(-1)
                    #     prefix_action_loss = result.action_loss[:, :len(cur_prefix) * 2 - 1].sum(-1)
                    #     prefix_loss = prefix_gpt_loss + prefix_action_loss
                    #     logger.info("prefix loss: {}".format(prefix_loss))
                    #     logger.info("cur_text_loss: {}".format(result.gpt_loss[0, len(cur_prefix):].sum(-1) + result.action_loss[0, len(cur_prefix) * 2 - 1:].sum(-1)))
                    
                    cur_loss = result.non_struct_loss.detach().cpu()
                # print(cur_loss.shape)
                # logger.info("prev_seq_len: {}".format(prev_seq_len))
                # logger.info("prefix loss: {}".format(cur_prefix_loss))
                # logger.info("cur_prefix_loss: {}".format(cur_loss[0, :prev_seq_len].sum(-1)))
                # logger.info("cur_text_loss: {}".format(cur_loss[0, prev_seq_len:].sum(-1)))
                prev_seq_len = seq_len - 1
                cur_loss -= cur_prefix_loss
                cur_doc_loss += _scalar(-torch.logsumexp(-cur_loss, dim=0))
                cur_prefix_loss = cur_loss.min().item() + cur_prefix_loss
                min_idx = cur_loss.argmin().item()
                cur_prefix = cur_seq[min_idx]
                logger.info(_scalar(-torch.logsumexp(-cur_loss, dim=0)) / cur_sent_len)
                cnt += 1

            if cur_doc_loss > 0:
                total_loss += cur_doc_loss

        logger.info('test ppl: {}'.format(np.exp(total_loss / total_token_length)))

    def evaluate_tg_doc(self, text_data, tg_seq_data, ranges, mask_rules):
        with open(text_data, 'r') as f:
            text_data = f.readlines()
        
        all_tg_seq = []
        cur_seq = []
        cur_seq_set = set()
        with open(tg_seq_data, 'r') as f:
            tg_seq_data = f.readlines()
            for i in  tqdm(range(len(tg_seq_data))):
                line = tg_seq_data[i].strip()
                if line.startswith('Sentence'):
                    if len(cur_seq) > 0:
                        all_tg_seq.append(cur_seq)
                        cur_seq = []
                        cur_seq_set = set()
                    continue
                if len(cur_seq) >= 300:
                    continue
                
                if line.strip() not in cur_seq_set:
                    cur_seq_set.add(line.strip())
                    seq = self.tokenizer.encode(line.strip())
                    cur_seq.append(seq)

            if len(cur_seq) > 0:
                all_tg_seq.append(cur_seq)
        
        cur_prefix = []
        cur_prefix_loss = 0
        total_loss = 0
        cur_doc_loss = 0
        total_token_length = 0
        cnt = 0
        self.model.eval()
        doc_num = 0
        prev_doc_num = -1
        prev_seq_len = 0
        fail_num = 0
        exclude = False
        with torch.no_grad():
            for i in tqdm(range(len(text_data))):
                if text_data[i].strip() == "":
                    logger.info("new doc")
                    cur_prefix = []
                    cur_prefix_loss = 0
                    total_loss += cur_doc_loss
                    cur_doc_loss = 0
                    doc_num += 1
                    prev_seq_len = 0
                    exclude = False
                    continue
                if exclude:
                    cnt += 1
                    continue

                # cur_text = self.tokenizer.encode(text_data[i].strip())
                # cur_sent_len = len(cur_text)
                
                cur_seq = all_tg_seq[cnt]
                cur_sent_len = sum([1 for t in cur_seq[0] if t < ranges.start_token])
                # if len(cur_text) != len(cur_parses[0]) + 1:
                #     logger.info(f'error: {len(cur_text)} != {len(cur_parses[0]) + 1}')
                #     logger.info(f'{cur_text}')
                #     logger.info(f'{cur_parses[0]}')
                #     exit()
                # if len(cur_text) * 2 - 1 != len(cur_seq[0]):
                #     logger.info(f'error: {len(cur_text) * 2 - 1} != {len(cur_seq[0])}')
                #     logger.info(f'{cur_text}')
                #     logger.info(f'{cur_seq[0]}')
                #     exit()  
                
                if cur_prefix != []:
                    cur_seq =[cur_prefix + seq for seq in cur_seq]
                seq_len = max([len(seq) for seq in cur_seq])

                if seq_len > 2047:
                    logger.info(f'exceed 2048 error: {seq_len}, {cnt}')
                    cnt += 1
                    if doc_num != prev_doc_num:
                        fail_num += 1
                        prev_doc_num = doc_num
                    exclude = True
                    continue

                seq_num = len(cur_seq)
                total_token_length += cur_sent_len
                cur_loss = None
                
                if (seq_len > 150 and seq_num > 20) or (seq_len > 100 and seq_num > 50):
                    for j in range(0, seq_num, 25):
                        chunk_input_ids = []
                        chunk_tgt_ids = []
                        chunk_masks = []
                        rel_positions = [] 
                        tmp_cur_seq = cur_seq[j:j+25]
                        inputs = {"chunk_input_ids": None, "chunk_tgt_ids": None}
                        
                        with ProcessPoolExecutor(max_workers=8) as executor:
                            results = executor.map(process_seq, tmp_cur_seq, [seq_len] * len(tmp_cur_seq))

                        for result in results:
                            src_p, tgt_p = result
                            chunk_input_ids.append(src_p)
                            chunk_tgt_ids.append(tgt_p)
                        # for seq in tmp_cur_seq:   
                        #     src_ = torch.LongTensor([ranges.start_token] + seq[:-1])
                        #     tgt_ = torch.LongTensor(seq)
                        #     info_tuple = masking_utils.compute_token_types(
                        #         {"inputs": src_, "labels": tgt_}, ranges
                        #     )
                        #     chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                        #                                             info_tuple['labels'], info_tuple['labels_ttypes'])

                        #     chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                        #     chunk = chunks[0]
                        #     src_p = chunk.inputs[:seq_len]
                        #     tgt_p = chunk.labels[:seq_len]
                        #     mask = chunk.attn_mask[:seq_len, :seq_len]
                        #     for i in range(len(mask)):
                        #         mask[i, i] = 1
                        #     chunk_len = len(chunk.inputs)
                        #     relpos = chunk.attn_relpos[:seq_len, chunk_len:chunk_len+seq_len]
                        #     chunk_input_ids.append(src_p)
                        #     chunk_tgt_ids.append(tgt_p)
                        #     chunk_masks.append(mask)
                        #     rel_positions.append(relpos)
                        
                        inputs['chunk_input_ids'] = torch.tensor(np.array(chunk_input_ids), dtype=torch.long).to(self.device)
                        inputs['chunk_tgt_ids'] = torch.tensor(np.array(chunk_tgt_ids), dtype=torch.long).to(self.device)
                        
                        with model.no_sync():
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                result = self.model(**inputs, tg_inference=True)
                        if cur_loss is None:
                            cur_loss = result.non_struct_loss.detach().cpu()
                        else:
                            cur_loss = torch.cat([cur_loss, result.non_struct_loss.detach().cpu()], dim=0)
                else:
                    inputs = {"chunk_input_ids": None, "chunk_tgt_ids": None}
                    chunk_input_ids = []
                    chunk_tgt_ids = []
                    chunk_masks = []
                    rel_positions = []
                    with ProcessPoolExecutor(max_workers=8) as executor:
                        results = executor.map(process_seq, cur_seq, [seq_len] * len(cur_seq))
                    
                    for result in results:
                        src_p, tgt_p = result
                        chunk_input_ids.append(src_p)
                        chunk_tgt_ids.append(tgt_p)
                    # for seq in cur_seq:
                    #     src = torch.LongTensor([ranges.start_token] + seq[:-1])
                    #     tgt = torch.LongTensor(seq)
                    #     info_tuple = masking_utils.compute_token_types(
                    #         {"inputs": src, "labels": tgt}, ranges
                    #     )
                    #     chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                    #                                             info_tuple['labels'], info_tuple['labels_ttypes'])
                    #     chunks = [types.Chunk(None, *chunk) for chunk in chunks]
                    #     chunk = chunks[0]
                    #     src_p = chunk.inputs[:seq_len]
                    #     tgt_p = chunk.labels[:seq_len]
                    #     mask = chunk.attn_mask[:seq_len, :seq_len]
                    #     for k in range(len(mask)):
                    #         mask[k, k] = 1
                    #     chunk_len = len(chunk.inputs)
                    #     relpos = chunk.attn_relpos[:seq_len, chunk_len:chunk_len+seq_len]
                    #     chunk_input_ids.append(src_p)
                    #     chunk_tgt_ids.append(tgt_p)
                    #     chunk_masks.append(mask)
                    #     rel_positions.append(relpos)
                    
                    inputs['chunk_input_ids'] = torch.tensor(np.array(chunk_input_ids), dtype=torch.long).to(self.device)
                    inputs['chunk_tgt_ids'] = torch.tensor(np.array(chunk_tgt_ids), dtype=torch.long).to(self.device)

                    with model.no_sync():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            result = self.model(**inputs, tg_inference=True)
                    # logger.info(result.action_loss)
                    # logger.info(result.gpt_loss)
                    # logger.info(cur_text)
                    # logger.info(len(cur_parses))
                    # if cur_prefix_loss != 0:
                    #     logger.info(cur_prefix_loss)
                    #     logger.info(len(cur_text) - len(cur_prefix))
                    #     prefix_gpt_loss = result.gpt_loss[:, :len(cur_prefix)].sum(-1)
                    #     prefix_action_loss = result.action_loss[:, :len(cur_prefix) * 2 - 1].sum(-1)
                    #     prefix_loss = prefix_gpt_loss + prefix_action_loss
                    #     logger.info("prefix loss: {}".format(prefix_loss))
                    #     logger.info("cur_text_loss: {}".format(result.gpt_loss[0, len(cur_prefix):].sum(-1) + result.action_loss[0, len(cur_prefix) * 2 - 1:].sum(-1)))
                    
                    cur_loss = result.non_struct_loss.detach().cpu()
                # print(cur_loss.shape)
                # logger.info("prev_seq_len: {}".format(prev_seq_len))
                # logger.info("prefix loss: {}".format(cur_prefix_loss))
                # logger.info("cur_prefix_loss: {}".format(cur_loss[0, :prev_seq_len].sum(-1)))
                # logger.info("cur_text_loss: {}".format(cur_loss[0, prev_seq_len:].sum(-1)))
                prev_seq_len = seq_len - 1
                cur_loss -= cur_prefix_loss
                cur_doc_loss += _scalar(-torch.logsumexp(-cur_loss, dim=0))
                cur_prefix_loss = cur_loss.min().item() + cur_prefix_loss
                min_idx = cur_loss.argmin().item()
                cur_prefix = cur_seq[min_idx]
                logger.info(_scalar(-torch.logsumexp(-cur_loss, dim=0)) / cur_sent_len)
                cnt += 1

            if cur_doc_loss > 0:
                total_loss += cur_doc_loss

        logger.info('test ppl: {}'.format(np.exp(total_loss / total_token_length)))


    def evaluate(self, text_data, parses, data_loader: DataLoader):
        with open(text_data, 'r') as f:
            text_data = f.readlines()
        text_data = [t.strip() for t in text_data]
        text_data = [self.tokenizer.encode(t) for t in tqdm(text_data, desc='tokenize')]
        all_parses = []
        cur_parses = []

        with open(parses, 'r') as f:
            parses = f.readlines()
            for line in tqdm(parses, desc='parse'):
                if line.startswith('Sentence'):
                    if len(cur_parses) > 0:
                        all_parses.append(cur_parses)
                        cur_parses = []
                    continue
                cur_parses.append(eval(line.strip()))
        if len(cur_parses) > 0:
            all_parses.append(cur_parses)
        
        assert len(text_data) == len(all_parses)

        self.model.eval()
        total_loss = 0
        total_token_length = 0
        total_struct_loss = 0
        total_non_struct_loss = 0
        total_gpt_loss = 0
        total_inside_outside_loss = 0
        total_parser_loss = 0
        total_action_loss = 0
        total_step = len(text_data)
        with torch.no_grad():
            for step, (cur_text, cur_parses) in enumerate(zip(text_data, all_parses)):
                # for k, v in inputs.items():
                #     if v is not None and isinstance(v, torch.Tensor):
                #         inputs[k] = v.to(self.device)
                if len(cur_text) != len(cur_parses[0]) + 1:
                    logger.info(f'error: {len(cur_text)} != {len(cur_parses[0]) + 1}')
                    logger.info(f'{cur_text}')
                    logger.info(f'{cur_parses[0]}')
                    continue
                # logger.info(f'{cur_text}')
                # logger.info(f'{cur_parses[0]}')
                # logger.info(len(cur_text))
                # logger.info(len(cur_parses[0]))
                inputs = {"chunk_input_ids": None, "chunk_masks": None, "input_ids": None, 
                          "masks": None, "group_ids": None, "chunk_parses": None}
                sent_len = len(cur_text)
                sent_num = len(cur_parses)
                chunk_input_ids = torch.tensor(cur_text, dtype=torch.long).repeat(sent_num, 1).view(-1, sent_len).to(self.device)
                chunk_masks = torch.ones_like(chunk_input_ids).long().to(self.device)
                input_ids = chunk_input_ids.view(-1, sent_len)
                masks = chunk_masks.view(-1, sent_len)
                group_ids = np.arange(sent_num)
                chunk_parses = torch.tensor(cur_parses, dtype=torch.long).to(self.device)
                inputs['chunk_input_ids'] = chunk_input_ids
                inputs['chunk_masks'] = chunk_masks
                inputs['input_ids'] = input_ids
                inputs['masks'] = masks
                inputs['group_ids'] = group_ids
                inputs['chunk_parses'] = chunk_parses
                result = self.model(**inputs, ppl=True)
            # for step, inputs in enumerate(data_loader):
            #     for k, v in inputs.items():
            #         if v is not None and isinstance(v, torch.Tensor):
            #             inputs[k] = v.to(self.device)
            #     with model.no_sync():
            #         amp_dtype=torch.float16
            #         with torch.cuda.amp.autocast(dtype=amp_dtype):
            #             result = self.model(**inputs)
                total_token_length += sent_len
                # logger.info(result.action_tgt.shape)
                # logger.info(result.action_tgt[0])
                # logger.info(result.gpt_loss.shape)
                # logger.info(result.action_loss.shape)
                # logger.info(result.action_loss[0])
                # logger.info(result.gpt_loss[0])
                cur_gpt_loss = result.gpt_loss.sum(-1)
                cur_action_loss = result.action_loss[:, :-1].sum(-1)
                # logger.info(result.action_loss.mean().mean().item())
                # logger.info(result.gpt_loss.mean().mean().item())
                cur_loss = cur_gpt_loss + cur_action_loss
                cur_ppl_loss = _scalar(-torch.logsumexp(-cur_loss, dim=0))
                logger.info(cur_ppl_loss / sent_len)
                total_loss += cur_ppl_loss
        logger.info('test ppl: {}'.format(np.exp(total_loss / total_token_length)))        
        return total_loss / total_step, total_struct_loss / total_step, total_non_struct_loss / total_step, \
               total_gpt_loss / total_step, total_inside_outside_loss / total_step, total_parser_loss / total_step, \
               total_action_loss / total_step

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--ext_vocab_path', required=False, default=None, type=str, help='external vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--accumulation_steps', type=int, default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2', 'r2d2-gen-fast', 'r2d2-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext', 'tg'], default='r2d2-gen')
    cmd.add_argument('--num_samples', type=int, default=100000)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff_start', type=float, default=1.0)
    cmd.add_argument('--coeff_end', type=float, default=0)
    cmd.add_argument('--coeff_proportion', type=float, default=0.8)
    cmd.add_argument('--temperature_start', type=float, default=1.0)
    cmd.add_argument('--temperature_end', type=float, default=0.1)
    cmd.add_argument('--temperature_proportion', type=float, default=0.8)
    cmd.add_argument('--pool_size', type=int, default=4)
    cmd.add_argument('--max_seq_len', type=int, default=2048)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--fix_embedding', action='store_true')
    cmd.add_argument('--disable_group', action='store_true')
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--gradient_checkpoint', action='store_true')
    # cmd.add_argument('--gpt_loss_coeff', type=float, default=1.0)
    cmd.add_argument('--compile', action='store_true')
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument('--eval_mode', type=int, default=0)
    cmd.add_argument('--dev_path', type=str, default='corpus/bllip')
    cmd.add_argument('--eval_steps', type=int, default=1000)
    cmd.add_argument('--train_sent', type=int, default=0)
    cmd.add_argument('--epoch', type=int, default=10)
    cmd.add_argument('--left', type=int, default=0)
    cmd.add_argument('--tg_enabled', type=int, default=0)

    args = cmd.parse_args(sys.argv[1:])
    torch.set_printoptions(profile='full')

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank
        while True:
            try:
                logging.info('init process group')
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                if torch.distributed.is_initialized():
                    break
            except ValueError:
                time.sleep(5)
            except:
                logging.error('Exit with unknown error')
                exit(-1)
        device = torch.device('cuda')
    else:
        global_rank = -1
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

    is_master = local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.mkdir(args.output_dir)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, 'training_log.txt'), mode='a', encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging
        
    logger.info(f'args: {args}')
    logger.info(f'initialize model on {global_rank}')
    gpt_config = AutoConfig.from_pretrained(args.gpt_config_path)
    bos_id = gpt_config.bos_token_id
    eos_id = gpt_config.eos_token_id
    model = create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path, args.fix_embedding, args.gradient_checkpoint)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        if args.model_type == 'tg':
            if is_master:
                state_dicts = torch.load(os.path.join(args.pretrain_dir, 'model.bin'), map_location=lambda a, b: a)
                out_dict = {}
                for key, val in state_dicts.items():
                    new_key = key.replace('module.gpt.', 'gpt.')
                    out_dict[new_key] = val
                model.load_state_dict(out_dict)
        else:
            model.from_pretrain(args.pretrain_dir, strict=False)
        logger.info("load from pretrain dir successfully")
    if args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.checkpoint_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            if args.model_type == 'gpt':
                if is_master:
                    state_dicts = torch.load(os.path.join(args.checkpoint_dir, f'model{max_epoch}_{max_step}.bin'), map_location=lambda a, b: a)
                    out_dict = {}
                    for key, val in state_dicts.items():
                        new_key = key.replace('module.gpt.', '')
                        out_dict[new_key] = val

                    torch.save(out_dict, os.path.join(args.checkpoint_dir, f'pytorch_model.bin'))
                torch.distributed.barrier()
                model.from_pretrain(args.checkpoint_dir)
            else:
                model.from_pretrain(os.path.join(args.checkpoint_dir, f'model{max_epoch}_{max_step}.bin'))
            # TODO: add loading from checkpoint for the parser
    
    logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)

    # named_par_list = list(model.named_parameters())
    # unused_parser_indices = "248 249"
    # unused_parser_indices = [int(t) for t in unused_parser_indices.split()]
    # for idx in unused_parser_indices:
    #     print(named_par_list[idx][0])

    set_seed(args.seed)

    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    if args.tg_enabled == 1:
        non_terminal_token = ['(ADJP', '(ADVP', '(CONJP', '(FRAG', '(INTJ', '(LST', '(NAC', '(NP', '(NX', '(PP', '(PRN', '(PRT', '(QP', '(RRC', '(S', '(SBAR', '(SBARQ', '(SINV', '(SQ', '(UCP', '(VP', '(WHADJP', '(WHADVP', '(WHNP', '(WHPP', '(X', \
                              'ADJP)', 'ADVP)', 'CONJP)', 'FRAG)', 'INTJ)', 'LST)', 'NAC)', 'NP)', 'NX)', 'PP)', 'PRN)', 'PRT)', 'QP)', 'RRC)', 'S)', 'SBAR)', 'SBARQ)', 'SINV)', 'SQ)', 'UCP)', 'VP)', 'WHADJP)', 'WHADVP)', 'WHNP)', 'WHPP)', 'X)']
        tokens = [AddedToken(token, lstrip=True) for token in non_terminal_token]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    print("vocab_size: ", len(tokenizer))
    ranges = None
    mask_rules = None
    if args.tg_enabled == 1:
        opening_nt = (tokenizer.get_vocab()['(ADJP'], tokenizer.get_vocab()['(X'] + 1)
        closing_nt = (tokenizer.get_vocab()['ADJP)'], tokenizer.get_vocab()['X)'] + 1)
        ranges = masking_utils.TokenTypeRanges(bos_id, -100, eos_id, opening_nt, closing_nt)
        mask_rules = masking_utils.get_masking_rules(
                "txl", 
                sequence_length=args.max_seq_len,
                memory_length=args.max_seq_len
        )
    
    lazy_loader = LazyLoader(args.corpus_path, transformer_grammar=bool(args.tg_enabled), is_array=True)

    dataset = GPT2Dataset(lazy_loader, num_samples=args.num_samples, tg=bool(args.tg_enabled), 
                          bos_id=bos_id, ranges=ranges, mask_rules=mask_rules, max_seq_len=args.max_seq_len - 1) # -1 for bos in mask rules
    print(f'total samples: {args.num_samples}')
    
    collator = DefaultCollator()
    collator_fn = collator.generative_gpt2_fn_tg_seq

    parser_params = []
    model_params = []
    for name, params in model.named_parameters():
        if name.find('.parser.') > 0:
            parser_params.append(params)
        else:
            model_params.append(params)
    if args.train_sent and args.tg_enabled == 0:
        with open("corpus/bllip/bllip_train.txt", 'r') as f:
            text_data = f.readlines()
        text_data = [t.strip() for t in text_data]
        # with ProcessPoolExecutor(max_workers=8) as executor:
        #     text_data = list(executor.map(self.tokenizer.encode, tqdm(text_data, desc='tokenize')))
        text_data = [tokenizer.encode(t) for t in tqdm(text_data, desc='tokenize')]
        parses = []
        if args.left == 0:
            with open("corpus/bllip/train_new_parses.txt", 'r') as f:
                parses = f.readlines()
                parses = [eval(p.strip()) for p in parses]
        else:
            with open("corpus/bllip_supar_train.lazy/data.parses.txt", 'r') as f:
                parses = f.readlines()
                parses = [eval(p.strip()) for p in parses]
        assert len(text_data) == len(parses)
        # for i in range(len(text_data)):
        #     if len(text_data[i]) != len(parses[i]) + 1:
        #         logger.info(f'error: {len(text_data[i])} != {len(parses[i]) + 1}')
        #         logger.info(f'{i}')
        # exit()
        total_step = (args.epoch * len(text_data)) // args.batch_size
        t_total = total_step
        warm_up_steps = args.warm_up * total_step
        optimizer = AdamW([{"params": model_params},
                            {"params": parser_params, "lr": args.parser_lr}],
                            lr=args.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                    num_training_steps=total_step // args.accumulation_steps)
        if global_rank >= 0:
            model = DDP(model)
        
    elif global_rank == -1:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        n_gpu = 1
        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total
        # TODO: seperate learning rate
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                    num_training_steps=t_total // args.accumulation_steps)
    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=False),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                    num_training_steps=t_total // args.accumulation_steps)
        model = DDP(model)
        
    coeff_scheduler = LinearProgressScheduler(args.coeff_start, args.coeff_end, args.coeff_proportion, t_total)
    temp_scheduler = LinearProgressScheduler(args.temperature_start, args.temperature_end, args.temperature_proportion, t_total)
    scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)

    # force setting base learning rate
    # scheduler.base_lrs = [args.lr, args.parser_lr]

    trainer = Trainer(model, collator, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master, num_workers=args.pool_size)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16
    # if args.eval_mode == 1:
    #     if args.left == 1:
    #         trainer.evaluate('corpus/bllip/binary_topk_300_text.txt', 'corpus/bllip/supar_eval_parses_new.txt', dataloader)
    #     else:
    #         trainer.evaluate('corpus/bllip/topks_3_text.txt', 'corpus/bllip/topks_3_new_parses.txt', dataloader)
    # elif args.eval_mode == 2:
    #     if args.left == 1:
    #         trainer.evaluate_doc('corpus/bllip/bllip_test.raw', 'corpus/bllip/supar_eval_parses_new.txt')
    #     else:
    #         trainer.evaluate_doc('corpus/bllip/bllip_test.raw', 'corpus/bllip/test_doc_level_parses.txt')

    def process_seq(seq, seq_len):
        src_ = torch.LongTensor([ranges.start_token] + seq)
        tgt_ = torch.LongTensor(seq + [ranges.pad_token])
        info_tuple = masking_utils.compute_token_types(
            {"inputs": src_, "labels": tgt_}, ranges
        )
        chunks = mask_rules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                                                info_tuple['labels'], info_tuple['labels_ttypes'])

        chunks = [types.Chunk(None, *chunk) for chunk in chunks]
        chunk = chunks[0]
        src_p = chunk.inputs[:seq_len+1]
        tgt_p = chunk.labels[:seq_len+1]
        return src_p, tgt_p

    if args.tg_enabled == 1 and args.eval_mode == 0:
        trainer.train_tg(dataloader, optimizer, scheduler, scaler,
                  args.output_dir,
                  amp_dtype=amp_dtype,
                  coeff_scheduler=coeff_scheduler,
                  temp_scheduler=temp_scheduler,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  accumulation_steps=args.accumulation_steps,
                  max_recover_step=max_step, eval_steps=args.eval_steps, dev_path=args.dev_path, ranges=ranges, mask_rules=mask_rules)
    elif args.eval_mode == 1 and is_master:
        if args.left == 1:
            trainer.evaluate('corpus/bllip/binary_topk_300_text.txt', 'corpus/bllip/gpst_internal_eval_topk.txt', dataloader, ranges, mask_rules)
    elif args.eval_mode == 2 and is_master:
            trainer.evaluate_doc('corpus/bllip/bllip_test.raw', 'corpus/bllip/gpst_internal_eval_topk.txt', ranges, mask_rules)
    elif args.train_sent:
        trainer.train_sent(text_data, parses, optimizer, scheduler, scaler,
                           args.output_dir, amp_dtype=amp_dtype,
                           coeff_scheduler=coeff_scheduler,
                           temp_scheduler=temp_scheduler,
                           epoch=args.epoch,
                           batch_size=args.batch_size,
                           log_steps=args.log_steps, save_steps=args.save_steps,
                           accumulation_steps=args.accumulation_steps,
                           eval_steps=args.eval_steps, dev_path=args.dev_path, left=args.left)
    else:
        trainer.train(dataloader, optimizer, scheduler, scaler,
                  args.output_dir,
                  amp_dtype=amp_dtype,
                  coeff_scheduler=coeff_scheduler,
                  temp_scheduler=temp_scheduler,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  accumulation_steps=args.accumulation_steps,
                  max_recover_step=max_step, eval_steps=args.eval_steps, dev_path=args.dev_path, left=args.left)