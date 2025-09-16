# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.model_loader import load_model
from datetime import datetime
from model.modeling_outputs import R2D2GenOutput
from concurrent.futures import ThreadPoolExecutor

'''
parallel-decoding
input format (training):
1. encoder_chunk_input_ids: (batch_size, raw_input_length)
   ex: torch.tensor([[1,[mask],3,4,[mask], 6,-100,-100],[2,4,6,[mask], 10,[mask],14,16]],dtype=torch.long) pad(-100) to fixed length(1024)
   post_process: will mask several positions then put into encoder and do mlm training
2. encoder_sentence_ids: (batch_size, raw_input_length)
   ex: torch.tensor([[0,0,0,0,0, 1,1,1],[0,0,0,0, 1,1,1,1]], dtype=torch.long) pad(last_sentence_id) to fixed length(1024), start from 0
3. encoder_padding_ids: (batch_size, raw_input_length)
   ex: torch.tensor([[0,0,0,0,0, 0,1,1],[0,0,0,0, 0,0,0,0]], dtype=torch.long) indentify pad positions 0: not pad; 1: pad
4. encoder_labels: (batch_size, raw_input_length)
   ex: torch.tensor([[-100,2,-100,-100,5, 6,-100,-100],[-100,-100,-100,8, -100,12,-100,-100]],dtype=torch.long) -100 to ignore

5. decoder_chunk_input_embeddings: (batch_size, tree_input_length, embed_dim) [in_fact: all paddings]
   ex: torch.tensor(dtype=torch.float)
6. decoder_sentence_ids: (batch_size, tree_input_length)
   ex: torch.tensor([[0,0,0, 1,1,1,1,1,1,1, 2,2,2,2],[0,0,0,0,0,0,0, 1,1,1,1,1,1,1]], dtype=torch.long) pad(last_sentence_id) to fixed length(2048), start from 0
7. decoder_padding_ids: (batch_size, tree_input_length)
   ex: torch.tensor([[0,0,0, 0,0,0,0,0,0,0, 0,0,0,1],[0,0,0,0,0,0,0, 0,0,0,0,0,0,0]], dtype=torch.long) indentify pad positions 0: not pad; 1: pad
8. decoder_labels: (batch_size, tree_input_length)
   ex: torch.tensor([[-1,0,1, -1,-1,-1,2,3,4,5, -1,6,7,-100],[-1,-1,-1,0,1,2,3, -1,-1,-1,4,5,6,7]], dtype=torch.long) -1: non-terminal position, -100: padding position 
9. depth_ids: (batch_size, tree_input_length)
   ex: torch.tensor([[0,1,1, 0,1,1,2,2,2,2, 0,1,1,1],[0,1,1,2,2,2,2, 0,1,1,2,2,2,2]], dtype=torch.long) pad(last_depth_id) to fixed length(2048), start from 0
10. decoder_embeds: (batch_size, tree_input_length, embed_dim)
   ex: torch.tensor(dtype=torch.float)

input format (inference)
'''


class FastGenerativeR2D2(nn.Module):
    def __init__(self, r2d2, action_layers, generation_layers, vocab_size, 
                 r2d2_input_dim, embedding_dim, dropout_rate=0.2, ext_vocab_size=0, 
                 fix_embeddings=False):
        # embedding dim is used to feed to r2d2
        # input dim is sued to feed to GPT
        super().__init__()
        self.embedding_dim = embedding_dim  # embedding_dim > r2d2_input_dim
        self.r2d2_input_dim = r2d2_input_dim
        self.r2d2 = r2d2

        self.vocab_size = vocab_size

        # self.action_ln = nn.Linear(self.embedding_dim, 2)  # judge reduce or predict next token

        self.enable_gpt = False
        if action_layers is not None:
            self.action_layers = action_layers
            self.bos_embedding = nn.Parameter(torch.rand(self.embedding_dim))
            self.up_scale = nn.Linear(self.r2d2_input_dim, self.embedding_dim)
            self.dense = nn.Sequential(nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(4 * self.embedding_dim, self.embedding_dim))
            self.action_mlp = nn.Sequential(nn.LayerNorm(self.embedding_dim),
                                nn.Linear(self.embedding_dim, self.embedding_dim),
                                nn.GELU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(self.embedding_dim, 2))
            self.enable_gpt = True
        
        self.classifier = nn.Linear(self.embedding_dim, vocab_size, bias=False)
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.embeddings.requires_grad = not fix_embeddings
        self.down_scale = nn.Linear(self.embedding_dim, self.r2d2_input_dim)

        self.insideoutside_dense = nn.Sequential(
            nn.Linear(r2d2_input_dim, 4 * r2d2_input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * r2d2_input_dim, self.embedding_dim)
        )

        # self.parallel_stream = torch.cuda.Stream()

        self._init_weights()
        self._tie_weights()

    def _init_weights(self):
        if self.enable_gpt:
            self.bos_embedding.data.normal_(mean=0, std=0.02)
        self.embeddings.weight.data.normal_(mean=0, std=0.02)

    def _tie_weights(self):
        self.classifier.weight = self.embeddings.weight

    # def get_parser(self):
    #     return self.r2d2.parser
        
    def from_pretrain(self, model_path, strict=True):
        load_model(self, model_path, strict=strict)
        self._tie_weights()

    def _append_eos_label(self, eos_labels, chunk_input_ids, chunk_masks, next_token_indices, max_input_len):
        # chunk_masks = (chunk_masks.sum(dim=1) > 0).to(int)
        seq_lens = (chunk_masks != 0).sum(dim=1)
        temp_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 1), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device)
        temp_ids.fill_(-100)
        temp_ids[:, :-1] = chunk_input_ids
        # comment this line to support discriminant way
        temp_ids.scatter_(1, seq_lens.unsqueeze(1), torch.tensor(eos_labels, device=chunk_input_ids.device).unsqueeze(1))
        chunk_input_ids = temp_ids
        next_token_indices = next_token_indices[:, :max_input_len + 1]
        return next_token_indices, chunk_input_ids

    def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, chunk_attn_masks=None, eos_labels=None, group_ids=None, chunk_parses=None, 
                atom_spans=None, span_ids=None, external_vocab_ids=None, 
                coeff=1.0, temperature=1.0, past_key_values=None, ppl=False, eval_dev=False, **kwargs):

        batch_size = max(group_ids) + 1
        r2d2_input_ids = torch.where(chunk_input_ids == -100, 0, chunk_input_ids)
        input_embeddings = self.embeddings(r2d2_input_ids)
        r2d2_embeddings = self.down_scale(input_embeddings)
        # max_input_len = chunk_input_ids.shape[1]

        # max_input_len = (chunk_masks != 0).sum(dim=1).max().to('cpu', non_blocking=True)
        # print((chunk_masks != 0).sum(dim=1).max().to('cpu', non_blocking=True))
        max_input_len = chunk_masks.shape[1]
        # print("max_input_len: ", max_input_len)
        # print("max_input_len: ", max_input_len)
        
        # ctx, outside_tgt, ldr_repr, position_ids, tgt_ids, token_indices, ext_ids, split_targets, l_height = \
        ctx, outside_tgt, ldr_repr, position_ids, tgt_ids, token_indices, ext_ids, l_height = \
            self.r2d2(r2d2_input_ids, chunk_masks, input_ids, masks, r2d2_embeddings, group_ids, chunk_parses, 
                      max_input_len, atom_spans=atom_spans, coeff=coeff, temperature=temperature, span_ids=span_ids,
                      eos_labels=eos_labels, external_vocab_ids=external_vocab_ids)


        if self.training:
            # with torch.cuda.stream(self.parallel_stream):
            # _loss = self.r2d2.parser_loss(ctx)
            outside_embeddings = self.r2d2.outside_embeddings(ctx)
            outside_logits = self.classifier(self.insideoutside_dense(outside_embeddings))
            insideoutside_loss = F.cross_entropy(outside_logits, outside_tgt)
        else:
            # parser_loss = insideoutside_loss = 0
            insideoutside_loss = 0
        
        logits = action_logits = None
        gpt_loss = action_loss = 0
        past_kv = None
        hidden_states = None

        if self.enable_gpt:
            if past_key_values is not None:
                action_past_kv, gen_past_kv = past_key_values
            else:
                action_past_kv = gen_past_kv = None
            gpt_input = self.up_scale(ldr_repr)
            gpt_input.scatter_(1, token_indices.unsqueeze(2).repeat(1, 1, input_embeddings.shape[-1]), 
                                input_embeddings.to(gpt_input.dtype))
            
            # ext_embedding = self.ext_embeds(ext_ids)
            # gpt_input = gpt_input + ext_embedding
            bos_emb = self.bos_embedding.unsqueeze(0).repeat(batch_size, 1)
            # position ids already considered <bos>
            cat_input = torch.cat([bos_emb.unsqueeze(1), gpt_input], dim=1)  # (group_size, L + 1, dim)
            # cat_input = self.layer_norm(cat_input)
            # cat_input = self.norm(cat_input)
            # print(len(cat_input[0]))
            # print(len(position_ids[0]))
            # print(position_ids[0])
            # print(chunk_attn_masks[0][1][:6])
            # print(chunk_attn_masks[0][2][:6])
            # print(chunk_attn_masks[0][3][:6])
            # print(chunk_attn_masks[0][4][:6])
            # print(chunk_attn_masks[0][5][:6]) 
            # exit()
            if chunk_attn_masks is not None:
                chunk_attn_masks = chunk_attn_masks[:cat_input.shape[0], :cat_input.shape[1], :cat_input.shape[1]]
            outputs = self.action_layers(inputs_embeds=cat_input, attention_mask=chunk_attn_masks, position_ids=position_ids, past_key_values=action_past_kv)  # (B, L, dim)
            action_logits = self.action_mlp(outputs.last_hidden_state)  # (B, L, 2)
            action_tgt = torch.where(tgt_ids == self.r2d2.reduce_id, 1, 0)  # REDUCE: 1, SHIFT:0
            action_tgt = torch.where(tgt_ids != -1, action_tgt, -1)
            # print(action_tgt)
                
            # print("tgt_ids: ", tgt_ids, tgt_ids.shape)
            # exit()
            next_token_indices = (tgt_ids != self.r2d2.reduce_id).int().argsort(dim=-1, descending=True, stable=True)  # (B, L)
            # print("next_token_indices: ", next_token_indices, next_token_indices.shape)
            if eos_labels is None:
                truncated_len = max_input_len if self.training or ppl or eval_dev else max_input_len + 1
                next_token_indices = next_token_indices[:, :truncated_len]
            else:
                next_token_indices, chunk_input_ids = self._append_eos_label(eos_labels, chunk_input_ids, chunk_masks, next_token_indices, max_input_len)
            token_outputs = outputs.last_hidden_state.gather(1, next_token_indices.unsqueeze(2).repeat(1, 1, self.embedding_dim))
            # print("token_outputs_shape: ", token_outputs.shape)
            # token_pos_ids = position_ids.gather(1, next_token_indices)
            # gather outputs to predict the next token
            # token_outputs = self.generation_layers(inputs_embeds=token_outputs, past_key_values=gen_past_kv)

            # hidden_states = token_outputs.last_hidden_state
            hidden_states = token_outputs

            logits = self.classifier(self.dense(hidden_states))  # (group_size, L + 1, vocab)
            # predict token loss + action loss
            # print("chunk_input_ids: ", chunk_input_ids)
            if self.training:
                # print("chunk_input_ids: ", chunk_input_ids)
                gpt_loss = F.cross_entropy(logits.permute(0, 2, 1), chunk_input_ids, ignore_index=-100)
                action_loss = F.cross_entropy(action_logits.permute(0, 2, 1), action_tgt, ignore_index=-1)
            elif ppl:
                gpt_loss = F.cross_entropy(logits.permute(0, 2, 1), chunk_input_ids, ignore_index=-100, reduction='none')
                action_loss = F.cross_entropy(action_logits.permute(0, 2, 1), action_tgt, ignore_index=-1, reduction='none')
            elif eval_dev:
                gpt_loss = F.cross_entropy(logits.permute(0, 2, 1), chunk_input_ids, ignore_index=-100)
                action_loss = F.cross_entropy(action_logits.permute(0, 2, 1), action_tgt, ignore_index=-1)
            # if gpt_loss is None or action_loss is None:
            #     gpt_loss = F.cross_entropy(logits.permute(0, 2, 1), chunk_input_ids, ignore_index=-100, reduction='none')
            #     action_loss = F.cross_entropy(action_logits.permute(0, 2, 1), action_tgt, ignore_index=-1, reduction='none')
            
            past_kv = (outputs.past_key_values)
        
        # torch.cuda.synchronize()
        # return loss + lm_loss + parser_loss, split_targets
        if ppl or eval_dev:
            return R2D2GenOutput(struct_loss=insideoutside_loss, 
                             # non_struct_loss=0.5 * gpt_loss + action_loss + parser_loss,
                             non_struct_loss=None, 
                             logits=logits,
                             action_logits=action_logits,
                             action_tgt=action_tgt,
                             tgt=chunk_input_ids,
                             hidden_states=hidden_states, 
                             tgt_ids=chunk_input_ids, 
                             gpt_loss=gpt_loss,
                             action_loss=action_loss,
                             inside_outside_loss=insideoutside_loss,
                             # parser_loss=parser_loss,
                             past_kv=past_kv)
                             # splits=split_targets)
        else:
            return R2D2GenOutput(struct_loss=insideoutside_loss, 
                             # non_struct_loss=0.5 * gpt_loss + action_loss + parser_loss,
                             non_struct_loss=0.5 * gpt_loss + action_loss, 
                             logits=logits,
                             action_logits=action_logits,
                             hidden_states=hidden_states, 
                             tgt_ids=chunk_input_ids, 
                             gpt_loss=gpt_loss,
                             action_loss=action_loss,
                             inside_outside_loss=insideoutside_loss,
                             # parser_loss=parser_loss,s
                             past_kv=past_kv)


class FastGenerativeR2D2_discriminant_glue(FastGenerativeR2D2):
    
    def _append_eos_label(self, eos_labels, chunk_input_ids, chunk_masks, next_token_indices, max_input_len):
        # chunk_masks = (chunk_masks.sum(dim=1) > 0).to(int)
        seq_lens = (chunk_masks != 0).sum(dim=1)
        temp_ids = torch.zeros((chunk_input_ids.shape[0], chunk_input_ids.shape[1] + 1), dtype=chunk_input_ids.dtype, device=chunk_input_ids.device)
        temp_ids.fill_(-100)
        temp_ids[:, :-1] = chunk_input_ids
        # temp_ids.scatter_(1, seq_lens.unsqueeze(1), torch.tensor(eos_labels, device=chunk_input_ids.device).unsqueeze(1))
        chunk_input_ids = temp_ids
        next_token_indices = next_token_indices[:, :max_input_len + 1]
        return next_token_indices, chunk_input_ids




'''
{
'chunk_input_ids': tensor([[  464, 10088,   290,  7693,  5847,   286, 45248,   290, 13091, 14447,
         12527,   531,   262,  4283,  6626,   318,   663,  1218,  2162,   340,
           550,   257,   513,    12,  1640,    12,    17,  6626,   287,  2805,
         12113,   764,   464, 45526,   837, 14248,    13,   837,  1664,   468,
           299,   470,  1683,  3432,   257,  5003, 30494,   764,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
        [   40,   423,   407,   587,  7559, 36777,   837, 10148,   290,   314,
          2740,   691,   329,  3589,   764,  3666,  2267,   837,   543,  1690,
          9018,  7734,   286,  5096,  6593,   314,  2074, 18992,   837,   318,
          5292,   329,   262,  4414,   286,   262,  1171,   764,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]]), 
'chunk_masks': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'input_ids': tensor([[  464, 10088,   290,  7693,  5847,   286, 45248,   290, 13091, 14447,
         12527,   531,   262,  4283,  6626,   318,   663,  1218,  2162,   340,
           550,   257,   513,    12,  1640,    12,    17,  6626,   287,  2805,
         12113,   764],
        [  464, 45526,   837, 14248,    13,   837,  1664,   468,   299,   470,
          1683,  3432,   257,  5003, 30494,   764,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [   40,   423,   407,   587,  7559, 36777,   837, 10148,   290,   314,
          2740,   691,   329,  3589,   764,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [ 3666,  2267,   837,   543,  1690,  9018,  7734,   286,  5096,  6593,
           314,  2074, 18992,   837,   318,  5292,   329,   262,  4414,   286,
           262,  1171,   764,     0,     0,     0,     0,     0,     0,     0,
             0,     0]]), 
'masks': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0]]), 
'group_ids': array([0, 0, 1, 1]), 
'chunk_parses': tensor([[31, 17, 10,  4,  0,  1,  2,  3,  5,  6,  7,  8,  9, 11, 14, 12, 13, 15,
         16, 18, 30, 19, 20, 27, 21, 26, 22, 23, 24, 25, 28, 29, 38, 32, 37, 33,
         34, 36, 35, 46, 39, 41, 40, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58],
        [14,  5,  0,  1,  2,  3,  4,  6,  7,  8, 13,  9, 10, 11, 12, 28, 16, 15,
         17, 27, 18, 19, 20, 21, 22, 24, 23, 25, 26, 36, 29, 30, 31, 33, 32, 34,
         35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58]]), 
'span_ids': [], 
'external_vocab_ids': None
}
'''