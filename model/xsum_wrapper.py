import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modeling_outputs import R2D2GenOutput

class XSumWrapper(nn.Module):
    # can be treated as xsum_generator
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def from_pretrain(self, model_path, strict=True):
        self.model.from_pretrain(model_path, strict=strict)

    # generative-r2d2
    # def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, eos_labels=None, group_ids=None, 
    #             atom_spans=None, span_ids=None, external_vocab_ids=None, 
    #             coeff=1.0, temperature=1.0, gpt_loss_coeff=1.0, past_key_values=None):
    # generative-r2d2-fast
    # def forward(self, chunk_input_ids= None, chunk_masks=None, input_ids=None, masks=None, eos_labels=None, group_ids=None, 
    #         atom_spans=None, span_ids=None, external_vocab_ids=None, 
    #         coeff=1.0, temperature=1.0, gpt_loss_coeff=1.0, past_key_values=None):
    # gptwrapper
    # def forward(self, chunk_input_ids= None, chunk_masks=None, input_ids=None, masks=None, group_ids=None, 
    #             max_input_len=0, atom_spans=None, enable_gpt=True, coeff=1.0, eos_labels=None, **kwargs):
    
    def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, summarys=None, eos_labels=None, group_ids=None, 
                atom_spans=None, span_ids=None, external_vocab_ids=None, 
                coeff=1.0, past_key_values=None, temperature=1.0):
        result = self.model.forward(chunk_input_ids=chunk_input_ids, chunk_masks=chunk_masks, input_ids=input_ids, masks=masks, eos_labels=eos_labels, group_ids=group_ids, 
                atom_spans=atom_spans, span_ids=span_ids, external_vocab_ids=external_vocab_ids, 
                coeff=coeff, temperature=temperature, past_key_values=past_key_values)
        return result
