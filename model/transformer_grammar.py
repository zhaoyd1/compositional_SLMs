import torch.nn as nn   
import torch
import torch.nn.functional as F
from utils.model_loader import load_model
from model.gpt2_flash_attn import GPT2LMHeadModel
from model.gpt2_flash_attn import GPT2Block
from model.modeling_outputs import R2D2GenOutput
from datetime import datetime
from filelock import FileLock
import os

class TG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpt = GPT2LMHeadModel(config)
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id
        self.embedding_dim = config.n_embd
        if hasattr(config, 'pointer_network') and config.pointer_network:
            self.pointer_network_Wq = nn.Linear(self.embedding_dim // 6, self.embedding_dim // 6)
            self.pointer_network_Wk = nn.Linear(self.embedding_dim // 6, self.embedding_dim // 6)
            for module in [self.pointer_network_Wq, self.pointer_network_Wk]:
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def from_pretrain(self, model_path, **kwargs):
        self.gpt = self.gpt.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
    
    def forward(self, chunk_input_ids=None, chunk_tgt_ids=None, attn_masks=None, pointer_mask=None, pointer_tgt_pos=None, rel_positions=None, input_ids=None, masks=None, group_ids=None, max_input_len=0, atom_spans=None, enable_gpt=True, coeff=1.0, eos_labels=None, tg_inference=False, **kwargs):
        if eos_labels is None:
            # seq_lens = (chunk_masks != 0).sum(dim=1)
            # tgt_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 1), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device).fill_(-100)
            # tgt_ids[:, 1:] = chunk_input_ids
            # tgt_ids[:, 0] = self.bos_id
            # gpt_input_ids = torch.where(tgt_ids != -100, tgt_ids, 0)
            tgt_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1]), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device).fill_(-100)
            tgt_ids[:, 1:] = chunk_tgt_ids[:, :-1]
            tgt_ids[:, 0] = self.bos_id
            tgt_ids = torch.where(tgt_ids == 0, -100, tgt_ids)
            # print(tgt_ids[0])
            # print(chunk_input_ids[0])
            result = self.gpt(input_ids=chunk_input_ids, labels=tgt_ids, attention_mask=attn_masks, return_dict=True, tg_inference=tg_inference)
            past_key_values = result.past_key_values
            pointer_loss = None
            pointer_logits = None
            if pointer_mask != None:
                pointer_Q = self.pointer_network_Wq(result.hidden_states[-1][..., :self.embedding_dim // 6])
                pointer_K = self.pointer_network_Wk(result.hidden_states[-1][..., :self.embedding_dim // 6])
                pointer_logits = torch.matmul(pointer_Q, pointer_K.transpose(-1, -2))
                pointer_logits = pointer_logits.masked_fill(~pointer_mask, torch.finfo(pointer_logits.dtype).min)
                # pointer_logits = F.softmax(pointer_logits, dim=-1)
                pointer_tgt_pos = torch.where(pointer_tgt_pos == 0, -100, pointer_tgt_pos)
                crit = nn.CrossEntropyLoss(ignore_index=-100)
                pointer_loss = crit(pointer_logits.view(-1, pointer_logits.size(-1)), pointer_tgt_pos.view(-1))
                # result.loss += pointer_loss
            
            return R2D2GenOutput(non_struct_loss=result.loss, pointer_loss=pointer_loss, struct_loss=None, logits=result.logits, pointer_logits=pointer_logits, hidden_states=result.hidden_states[-1], tgt_ids=chunk_tgt_ids, splits=None, past_kv=past_key_values)
        # else:
        #     seq_lens = (chunk_masks != 0).sum(dim=1)
        #     tgt_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 2), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device)
        #     tgt_ids.fill_(-100)
        #     tgt_ids[:, 1:-1] = chunk_input_ids
        #     tgt_ids[:, 0] = self.bos_id
        #     tgt_ids.scatter_(1, seq_lens.unsqueeze(1), torch.tensor(eos_labels, device=chunk_input_ids.device).unsqueeze(1))
        #     gpt_input_ids = torch.where(tgt_ids != -100, tgt_ids, 0)
        #     result = self.gpt(input_ids=gpt_input_ids, labels=tgt_ids, return_dict=True)
        #     return R2D2GenOutput(non_struct_loss=result.loss, struct_loss=0, logits=result.logits, splits=None)