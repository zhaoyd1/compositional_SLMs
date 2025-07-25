# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by chao lou & yida zhao


"""Masks for Transformer Grammars models."""

import dataclasses
import logging
from lib2to3.pgen2.tokenize import tokenize
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import torch


from . import constants as mc
from . import cpp_masking as mcpp


@dataclasses.dataclass(frozen=True)
class TokenTypeRanges:
    """Mapping between token IDs ranges to token types."""

    start_token: int
    pad_token: int
    end_token: int
    opening_non_terminals: Tuple[int, int]
    closing_non_terminals: Tuple[int, int]
    startofword: List[bool]

    def token_type_from_token(self, seq, *, tokens_type='inputs', use_pytorch=False):
        """Returns an array of token types from an array of token IDs."""
        if use_pytorch:
            np_ = torch
            dtype = torch.int32
        else:
            np_ = np
            dtype = np.int32

        start_token_mask = seq == self.start_token
        pad_token_mask = seq == self.pad_token
        end_token_mask = seq == self.end_token
        opening_nt_mask = (seq >= self.opening_non_terminals[0]) & (
            seq < self.opening_non_terminals[1]
        )
        closing_nt_mask = (seq >= self.closing_non_terminals[0]) & (
            seq < self.closing_non_terminals[1]
        ) 
        sow_mask = (np_.BoolTensor(self.startofword)[seq]) & (seq > 0)

        result = np_.full_like(seq, mc.TERMINAL, dtype=dtype)
        result[start_token_mask] = mc.SOS
        result[pad_token_mask] = mc.PAD
        result[opening_nt_mask] = mc.OPENING_NT
        result[closing_nt_mask] = mc.CLOSING_NT
        result[sow_mask] = mc.STARTOFWORD
        
        # if tokens_type == 'labels' and len(result) > 1:
        #     result[1] = mc.STARTOFWORD
        # elif len(result) >= 1:
        #     result[0] = mc.STARTOFWORD

        return result


def get_masking_rules(name, **kwargs):
    """Returns the masking rules instance."""
    if name == "stack_compose_double_closing_nt":
        # kwargs:
        #   int sequence_length
        #   int memory_length
        #   float transparency_prob
        #   bool use_relative_positions
        #   bool gather_into_new_memory: smart memory
        #   int transparency_depth_threshold:
        #           Depth below or at which the node is transparent
        #           -1 means that it's never transparent.
        #           <s> has depth 0, (DOC depth 1, so for the top level (S
        #           to be transparent, we need this to be set to 2
        cls = mcpp.StackComposeDoubleClosingNT
    elif name == "txl":
        # kwargs:
        #   int sequence_len
        #   int memory_len
        cls = mcpp.TXLCausalMasking
    else:
        raise NotImplementedError
    if kwargs is None:
        kwargs = dict()
    maskrules = cls(**kwargs)
    return maskrules


def compute_token_types(
    inp: Dict[str, np.ndarray], ranges: TokenTypeRanges
) -> Dict[str, np.ndarray]:
    """Computes token types using a dictionary."""
    for key in ("inputs", "labels"):
        if ranges is not None:
            inp[f"{key}_ttypes"] = ranges.token_type_from_token(inp[key], tokens_type=key, use_pytorch=True)
        else:
            inp[f"{key}_ttypes"] = np.zeros_like(inp[key])
    return inp