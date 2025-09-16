from enum import Enum
from typing import List
import torch
from copy import deepcopy

class ActionType(Enum):
    SHIFT = 0
    REDUCE = 1
    REDUCE2 = 3


# class Node:
#     def __init__(self, left, right, owner, terminal_id=-1, token_id=-1) -> None:
#         self.left = left
#         self.right = right
#         self.terminal_id = terminal_id
#         self.token_id = token_id
#         self.owner = owner

#         if self.left is not None and self.right is not None:
#             self.i = self.left.i
#             self.j = self.right.j
#         else:
#             self.i = terminal_id
#             self.j = terminal_id

#         if token_id != -1:
#             self.expr = str(token_id)
#         elif self.left is not None and self.right is not None:
#             self.expr = f'{self.left.expr},{self.right.expr}'
#         else:
#             self.expr = ''
    
#     def __repr__(self):
#         if self.terminal_id != -1:
#             return f'{self.terminal_id}'
#         else:
#             assert self.left is not None and self.right is not None
#             return f'({self.left}, {self.right})'

#     def to_ids(self):
#         if self.left is None and self.right is None:
#             return f'{self.token_id}'
#         else:
#             return f'({self.left.to_ids()} {self.right.to_ids()})'

#     def to_tokens(self, vocab):
#         if self.left is None and self.right is None:
#             return f'{vocab[self.token_id]}'
#         else:
#             return f'({self.left.to_tokens(vocab)} {self.right.to_tokens(vocab)})'

# def shift_conflict_with_atom_spans(i, j, atom_spans):
#     for atom_i, atom_j in atom_spans:
#         if j == atom_j and i > atom_i:
#             # shift before completing reduce action for an atom span is not allowed
#             return True
#     return False

# def reduce_conflict_with_atom_spans(i, j, atom_spans):
#     for atom_i, atom_j in atom_spans:
#         if i < atom_i <= j < atom_j or atom_i < i <= atom_j < j:
#             # reduce break any atom span is not allowed
#             return True
#     return False

INIT_VAL = 0.0

class State:
    def __init__(self, prev_state = None) -> None:
        self.prev_state = prev_state
        # self.prev_shift = None
        self.max_step = 0 if prev_state is None else prev_state.max_step
        self.total_len = prev_state.total_len if prev_state is not None else 0
        self.current_action = ActionType.SHIFT
        self.current_step = 0 if prev_state is None else prev_state.current_step + 1
        # self.stack_top = None  # reference to the node on the top of the stack
        self.input_offset = 0 if prev_state is None else prev_state.input_offset  # the index to the next token to shift
        self.invalid_state = False
        self.updated = False
        self.score = 0
        self.beam_id = -1
        self.batch_idx = -1
        self.stack = prev_state.stack[:] if prev_state is not None else []
        # TODO: modify the max_succ_open_nt to 5 for xsum
        self.all_closing_nt_cnt = prev_state.all_closing_nt_cnt if prev_state is not None else 0 # the number of all reduce
        self.composed_positions = set() # the positions of composed nodes
        self.current_seq = [] # the current sequence of actions
        self.token_id = -1 # must be changed in other places
        self.cur_composition_positions = set() # used for reduce as mask
        self.reduce2_pos_seq = [] # the positions of reduce2
    # def top_states(self, ext_vocab=None):
    #     if self.invalid_state:
    #         return None, None, 0
    #     ext_vocab_id = -1
    #     if ext_vocab is not None:
    #         if self.prev_shift is not None and self.prev_shift.stack_top is not None:
    #             assert self.stack_top is not None
    #             new_expr = f'{self.prev_shift.stack_top.expr},{self.stack_top.expr}'
    #             ext_vocab_id = ext_vocab.get(new_expr, -1)
    #     return self.prev_shift, self, ext_vocab_id + 1
    
    @property
    def token_offset(self):
        return min(self.input_offset, self.total_len)

    @property
    def is_finished(self):
        if self.token_offset == self.total_len and self.all_closing_nt_cnt == self.total_len - 1:
            return True # all tokens are shifted and all nts are reduced
        return False
    
    def action_masks(self, atom_spans=None):
        # TODO: For top-down, not allowing successive opening nts over M time steps; 
        #                     not allowing closing nts only when the stack is empty. 
        # TODO: For 
        if self.current_step >= self.max_step - 2: # do not allow to step into the next token
            self.invalid_state = True

        if self.invalid_state:
            return [False, False, False]

        # is shift valid?
        shift_valid = False
        reduce_valid = False
        reduce2_valid = False

        if self.current_action == ActionType.REDUCE:
            reduce2_valid = True
            return [shift_valid, reduce_valid, reduce2_valid]

        if self.input_offset < self.total_len:
            shift_valid = True
            # if atom_spans is not None and self.stack_top is not None:
            #     span_i, span_j = self.stack_top.i, self.stack_top.j
            #     if shift_conflict_with_atom_spans(span_i, span_j, atom_spans):
            #         shift_valid = False
        
        if self.all_closing_nt_cnt < self.total_len - 1 and len(self.stack) >= 2:
            reduce_valid = True

            # if atom_spans is not None:
            #     span_i, span_j = self.prev_shift.stack_top.i, self.stack_top.j
            #     if reduce_conflict_with_atom_spans(span_i, span_j, atom_spans):
            #         reduce_valid = False
        return [shift_valid, reduce_valid, reduce2_valid]
            
    def act(self, action: ActionType, token_id = -1):
        # action == the prediction of current state -> actiontype of next state
        # TODO: For top-down, one more type, opening a non-terminal. compose prev_shift util prev_shift is a OPEN
        #           tg-based -> one more type, REDUCE 2. For REDUCE 2, it is a duplicate of current state. (only ActionType changes)
        #           pointer-network -> record the idx in seq for each state !!! so we can find prev_shift until the seq_idx == what pointer is
        #           update the position_ids (maybe just the token offset) for 1000/1001, op nt -> 0, can just use token offset
        #                                                                 for other op nt, maintain position_ids
        # TODO: mask !!! each state keep a mask, if current state is a REDUCE 2, mask it out always.
        # self.prev_succ_open_nt = 0 # the number of successive open nt
        # self.max_succ_open_nt = 5 # the max number of successive open nt
        # self.uncomposed_open_nt = 0 # the number of uncomposed open nt
        # self.composed_positions = prev_state.composed_positions if prev_state is not None else [] # the positions of composed nodes
        # self.current_seq = prev_state.current_seq if prev_state is not None else [] # the current sequence of actions
        # self.token_id = -1 # must be changed in other places
        # self.open_nt_pos_seq = prev_state.open_nt_pos_seq if prev_state is not None else [] # the positions of open nt
        # self.cur_composition_positions = set()
        prev_reduce2 = False
        if self.current_action == ActionType.REDUCE2:
            prev_reduce2 = True
        
        if action == ActionType.SHIFT:
            # assert self.input_offset < self.total_len, f'input offset : {self.input_offset}, total len: {self.total_len}'
            if self.input_offset >= self.total_len or self.invalid_state or \
            self.current_action == ActionType.REDUCE:
                # nothing to shift, invalid state
                next_state = State(self)
                next_state.invalid_state = True
                next_state.input_offset = self.total_len - 1
                next_state.token_id = token_id
                if prev_reduce2:
                    next_state.composed_positions = self.composed_positions | set({self.current_step})
                    next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
                return next_state 
            next_state = State(self)
            next_state.reduce2_pos_seq = self.reduce2_pos_seq[:]
            if prev_reduce2:
                next_state.composed_positions = self.composed_positions | set({self.current_step})
                next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
            next_state.composed_positions = self.composed_positions | next_state.composed_positions
            next_state.current_action = ActionType.SHIFT
            next_state.stack = self.stack + [next_state.current_step]
            next_state.input_offset = self.input_offset + 1
            next_state.token_id = token_id
            next_state.current_seq = self.current_seq + [token_id]
            return next_state
        
        elif action == ActionType.REDUCE:
            # assert self.stack_depth >= 2
            if self.invalid_state or self.current_action == ActionType.REDUCE or len(self.stack) < 2 \
            or self.all_closing_nt_cnt >= self.total_len - 1:
                # nothing to reduce, invalid state
                next_state = State(self)
                next_state.invalid_state = True
                next_state.token_id = token_id
                next_state.input_offset = self.total_len - 1
                if prev_reduce2:
                    next_state.composed_positions = self.composed_positions | set({self.current_step})
                    next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
                return next_state
            next_state = State(self)
            next_state.reduce2_pos_seq = self.reduce2_pos_seq[:]
            if prev_reduce2:
                next_state.composed_positions = self.composed_positions | set({self.current_step})
                next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
            next_state.current_action = ActionType.REDUCE
            next_state.all_closing_nt_cnt += 1
            left_pos = self.stack[-2]
            right_pos = self.stack[-1]
            next_state.stack = self.stack[:-2] + [next_state.current_step]
            next_state.composed_positions = self.composed_positions | next_state.composed_positions # to deal with prev_reduce2, so that cur_compose can use next_state.composed_positions
            next_state.cur_composition_positions = set({left_pos, right_pos})
            next_state.composed_positions = set({left_pos, right_pos}) | next_state.composed_positions # the positions of composed nodes
            next_state.current_seq = self.current_seq + [token_id]
            next_state.token_id = token_id
            return next_state
        
        elif action == ActionType.REDUCE2:
            if self.invalid_state or self.current_action != ActionType.REDUCE:
                next_state = State(self)
                next_state.invalid_state = True
                next_state.token_id = token_id
                next_state.input_offset = self.total_len - 1
                if prev_reduce2:
                    next_state.composed_positions = self.composed_positions | set({self.current_step})
                    next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
                return next_state
            next_state = State(self)
            next_state.reduce2_pos_seq = self.reduce2_pos_seq[:]
            if prev_reduce2:
                next_state.composed_positions = self.composed_positions | set({self.current_step})
                next_state.reduce2_pos_seq = self.reduce2_pos_seq + [self.current_step]
            next_state.composed_positions = self.composed_positions | next_state.composed_positions
            next_state.current_action = ActionType.REDUCE2
            next_state.token_id = token_id
            next_state.current_seq = self.current_seq + [token_id]
            return next_state
        else:
            raise Exception('Unsupported action type!')

    def to_ids(self):
        return " ".join(map(str, self.current_seq))
    # def __repr__(self):
    #     if self.invalid_state:
    #         return 'invalid state'
    #     if self.prev_state is not None:
    #         act_name = f'R' if self.current_action == ActionType.REDUCE else f'S'
    #         return f'{self.prev_state} {act_name}'
    #     return ''

    # def to_ids(self):
    #     if self.current_action == ActionType.REDUCE or self.prev_shift is None:
    #         current_expr = ''
    #         if self.stack_top is not None:
    #             current_expr = self.stack_top.to_ids()
    #         if self.prev_shift is not None:
    #             return f'{self.prev_shift.to_ids()}, {current_expr}'
    #         else:
    #             return current_expr
    #     else:
    #         assert self.stack_top is not None
    #         if self.prev_shift is not None:
    #             return f'{self.prev_shift.to_ids()}, {self.stack_top.to_ids()}'
    #         else:
    #             return f'{self.stack_top.to_ids()}'

    # def to_tokens(self, vocab):
    #     if self.current_action == ActionType.REDUCE or self.prev_shift is None:
    #         current_expr = ''
    #         if self.stack_top is not None:
    #             current_expr = self.stack_top.to_tokens(vocab)
    #         if self.prev_shift is not None:
    #             return f'{self.prev_shift.to_tokens(vocab)}, {current_expr}'
    #         else:
    #             return current_expr
    #     else:
    #         assert self.stack_top is not None
    #         if self.prev_shift is not None:
    #             return f'{self.prev_shift.to_tokens(vocab)}, {self.stack_top.to_tokens(vocab)}'
    #         else:
    #             return f'{self.stack_top.to_tokens(vocab)}'


class BeamContext:
    def __init__(self, model, batch_size, max_beam_size, max_input_len, config, device, 
                 action_kv_history_len=0):
        self.device = device
        # self.embeddings = model.embeddings
        # self.up_scale = model.up_scale
        # self.down_scale = model.down_scale
        # self.layer_norm = model.layer_norm
        self.max_beam_size = max_beam_size
        # self.compose_cache = torch.full((batch_size, max_beam_size, max_input_len * 2, compose_dim), fill_value=0.0, device=device)
        # assert input_dim % num_head
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.action_layer_num = config.action_layer_num
        self.action_kv_history_len = action_kv_history_len
        self.prefix_token_lens = [0] * batch_size
        self.action_key_values = [(torch.full((batch_size, max_beam_size, self.head_num, 3 + 3 * max_input_len + action_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device),
                                   torch.full((batch_size, max_beam_size, self.head_num, 3 + 3 * max_input_len + action_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device)) for _ in range(self.action_layer_num)]
        self.action_masks = torch.full((batch_size, max_beam_size, 3 * max_input_len + action_kv_history_len + 4), fill_value=False, dtype=torch.bool, device=device)
        self.action_masks[:, :, -1] = True
        self.gpt_input_cache = torch.full((batch_size, max_beam_size), fill_value=0, dtype=torch.long, device=device) # directly use the token id

    def init_history_kv(self, action_key_values, seq_lens, group_ids):
        for layer_i, (k, v) in enumerate(self.action_key_values):
            k[:, 0, :, :self.action_kv_history_len, :] = action_key_values[layer_i][0]
            v[:, 0, :, :self.action_kv_history_len, :] = action_key_values[layer_i][1]
        # set masks
        N = max(group_ids) + 1
        action_len_bucket = [1] * N
        token_len_bucket = [1] * N
        for sent_i, group_id in enumerate(group_ids):
            action_len_bucket[group_id] += 2 * seq_lens[sent_i] - 1
            token_len_bucket[group_id] += seq_lens[sent_i]
        self.prefix_token_lens = [l - 1 for l in token_len_bucket]
        for batch_i in range(N):
            self.action_masks[batch_i, 0, :action_len_bucket[batch_i]] = True
        return action_len_bucket
        
    # def prepare_compositions(self, states_batch, beam_context):
    #     composition_indices = []
    #     for batch_i, states in enumerate(states_batch):
    #         for state in states:
    #             if state.current_action == ActionType.REDUCE and not state.is_finished:
    #                 l_top, r_top, ext_vocab_id = state.prev_state.top_states()

    #                 assert l_top is not None
    #                 assert r_top is not None
    #                 left_idx = l_top.current_step
    #                 right_idx = r_top.current_step

    #                 composition_indices.append((batch_i, state.beam_id, left_idx, right_idx))

    #     if len(composition_indices) > 0:
    #         composition_indices = torch.tensor(composition_indices, dtype=torch.long, device=self.device)
    #         left_repr = self.compose_cache[composition_indices[:, 0], composition_indices[:, 1], composition_indices[:, 2], :] # (?, dim)
    #         right_repr = self.compose_cache[composition_indices[:, 0], composition_indices[:, 1], composition_indices[:, 3], :] # (?, dim)
    #         return torch.stack([left_repr, right_repr], dim=1)  # (?, 2, dim)
    #     else:
    #         return None

    def update(self, states_batch: List[List[State]], sync_steps: List[int],  
               last_action_kv):
        force_sync = [False] * len(states_batch)
        beam_size = max(map(len, states_batch))
        org_beam_ids = []
        next_states_token_indices = []
        next_token_ids = []
        action_kv_indices = []
        action_kv_reorder_indices = []
        for batch_i, states in enumerate(states_batch):
            beam_ids = []
            tmp_force_sync = True
            for new_beam_id, state in enumerate(states):
                beam_ids.append(state.beam_id) # beam id of the previous state
                state.beam_id = new_beam_id
                # if state.current_action == ActionType.REDUCE and not state.is_finished:
                #     assert state.current_step > 2
                #     reduce_states_indices.append((batch_i, new_beam_id, state.current_step))
                if not state.updated:
                    # ignore those already generated next tokens
                    state.updated = True
                    tmp_force_sync = False
                    assert state.batch_idx != -1
                    # if state.current_action == ActionType.SHIFT:
                    #     # only update token kv for shift states
                    #     token_kv_indices.append((batch_i, new_beam_id, 
                    #         state.token_offset - 1 + self.token_kv_history_len))
                    #     token_kv_reorder_indices.append(state.batch_idx)
                    action_kv_indices.append((batch_i, new_beam_id, 
                        state.current_step + self.action_kv_history_len))
                    action_kv_reorder_indices.append(state.batch_idx)
                    # assert state.stack_top.token_id != -1
                    # NOTE: Will current_step exceed the max length? 
                    next_states_token_indices.append((batch_i, new_beam_id, state.current_step))
                    next_token_ids.append(state.token_id)
            force_sync[batch_i] = tmp_force_sync
            while len(beam_ids) < self.max_beam_size:
                beam_ids.append(self.max_beam_size - 1)

            org_beam_ids.append(beam_ids)

        # if reduce_repr is not None:
        #     assert len(reduce_states_indices) == reduce_repr.shape[0]
        # else:
        #     assert len(reduce_states_indices) == 0

        # reorder all caches
        org_beam_ids = torch.tensor(org_beam_ids, dtype=torch.long, device=self.device)
        action_kv_indices = torch.tensor(action_kv_indices, dtype=torch.long).to(self.device, non_blocking=True)
        assert org_beam_ids.shape[-1] == self.max_beam_size
        # L = self.compose_cache.shape[2]
        # self.compose_cache = self.compose_cache.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3)\
        #     .repeat(1, 1, L, self.compose_dim)) # B * max_beam_size * L * dim (choosing the previous beam_id)
        # (batch_size, max_beam_size, num_head, L, head_dim)
        L = self.action_key_values[0][0].shape[3]
        self.action_key_values = [(k.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim)),
                                  v.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim))) for k,v in self.action_key_values]
        L = self.action_masks.shape[-1]
        self.action_masks = self.action_masks.gather(dim=1, index=org_beam_ids.unsqueeze(-1).repeat(1, 1, L))
        # D = self.gpt_input_cache.shape[-1]
        self.gpt_input_cache = self.gpt_input_cache.gather(dim=1, index=org_beam_ids) # (batch_size, max_beam_size)

        # fillin compositional representations
        # if len(reduce_states_indices) > 0:
        #     reduce_states_indices = torch.tensor(reduce_states_indices, device=self.device, dtype=torch.long)
        #     assert not torch.all(self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2] - 1, :] == 0)
        #     assert torch.all(self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2], :] == 0)
        #     self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2], :] = reduce_repr
        #     self.gpt_input_cache[reduce_states_indices[:, 0], reduce_states_indices[:, 1], :] = self.up_scale(reduce_repr)
        
        # fillin action_key_values
        
            # print([state.current_step for state in states_batch[0]])

            # print([state.updated for state in states_batch[0]])
            
            # print([state.token_offset for state in states_batch[0]])

            # print([state.current_step for state in self.prev_states[0]])

            # print([state.updated for state in self.prev_states[0]])

            # print([state.token_offset for state in self.prev_states[0]])

            # breakpoint()
        # assert action_kv_indices.shape[0] > 0
        if action_kv_indices.shape[0] == 0:
            assert force_sync == [True] * len(states_batch)
    
        elif last_action_kv is not None:
            action_kv_reorder_indices = torch.tensor(action_kv_reorder_indices, dtype=torch.long, device=self.device)
            for layer_i, kv in enumerate(self.action_key_values):
                # if torch.all(action_kv_indices[:, 2] - 2 - self.action_kv_history_len >= 0):
                #     print(action_kv_indices[:, 2] - 2 - self.action_kv_history_len)
                #     assert torch.any(kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 2, :] != INIT_VAL)
                assert torch.all(kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] == INIT_VAL)
                # print(f'tgt shape {kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :].shape}')
                # print(f'src shape {last_action_kv[layer_i][0].shape}')
                kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] = \
                    last_action_kv[layer_i][0].index_select(dim=0, index=action_kv_reorder_indices).squeeze(-2)
                kv[1][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] = \
                    last_action_kv[layer_i][1].index_select(dim=0, index=action_kv_reorder_indices).squeeze(-2)
            # NOTE: check if reduce_state_indices[:,2] or reduce_state_indices[:,2]-1
            if torch.all(action_kv_indices[:, 2] - 1 >= 0):
                # assert torch.all(self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1])
                assert torch.all(~self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1])
            # NOTE: why -1 ? ignore the current action
            self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1] = True
        
        
        # print(self.action_masks[0, :5, :5])
        # print(self.action_key_values[0][0][0, :5, 0, :5, :5])

        # fillin token kv into shift ones
        
        next_token_ids = torch.tensor(next_token_ids, dtype=torch.long, device=self.device)
        # token_repr = self.embeddings(next_token_ids)
        if len(next_states_token_indices) > 0:
            next_states_token_indices = torch.tensor(next_states_token_indices, dtype=torch.long, device=self.device)
            # self.compose_cache[next_states_token_indices[:, 0], next_states_token_indices[:, 1], next_states_token_indices[:, 2], :] = self.down_scale(token_repr)
            self.gpt_input_cache[next_states_token_indices[:, 0], next_states_token_indices[:, 1]] = next_token_ids

        # self.prev_states = deepcopy(states_batch)
        return force_sync

    def prepare_gpt_input(self, states_batch: List[List[State]], sync_steps: List[int]):
        gpt_input_indices = []
        kv_indices = []
        position_ids = []
        tg_masks = []
        action_mask_len = self.action_masks.shape[-1]
        for batch_i, states in enumerate(states_batch):
            for state_i, state in enumerate(states):
                # assert state.beam_id == state_i
                if state.token_offset == sync_steps[batch_i] and not state.is_finished:
                    gpt_input_indices.append((batch_i, state.beam_id))
                    # TODO: modify the position_ids
                    # position_ids.append(state.token_offset + self.prefix_token_lens[batch_i])
                    position_ids.append(state.current_step + self.prefix_token_lens[batch_i])
                    kv_indices.append((batch_i, state.beam_id))
                    if state.current_action == ActionType.REDUCE:
                        cur_mask = torch.full((action_mask_len,), fill_value=False, dtype=torch.bool, device=self.device)
                        # # do not mask out the history
                        offset = torch.tensor(list(state.cur_composition_positions), dtype=torch.long, device=self.device) + self.action_kv_history_len
                        cur_mask[offset] = True
                        cur_mask[-1] = True # -1 is the current action
                        tg_masks.append(cur_mask)
                    else:
                        # cur_mask = torch.full((action_mask_len,), fill_value=True, dtype=torch.bool, device=self.device)
                        # offset = torch.tensor(state.reduce2_pos_seq, dtype=torch.long, device=self.device) + self.action_kv_history_len
                        # cur_mask[offset] = False
                        # tg_masks.append(cur_mask)
                        cur_mask = torch.full((action_mask_len,), fill_value=True, dtype=torch.bool, device=self.device)
                        offset = torch.tensor(list(state.composed_positions), dtype=torch.long, device=self.device) + self.action_kv_history_len
                        cur_mask[offset] = False
                        cur_mask[-1] = True
                        tg_masks.append(cur_mask)
        
        # if sync_steps[0] == 7:
        #     print([state.uncomposed_open_nt for state in states])
        #     print([state.current_action for state in states])
        #     print([state.current_step for state in states])
        #     print(states_batch[0][0].current_seq)
        #     print(tg_masks[0])
        #     breakpoint()
        if len(gpt_input_indices) > 0:
            tg_masks = torch.stack(tg_masks, dim=0)
            assert tg_masks.shape == (len(gpt_input_indices), action_mask_len)

        if len(gpt_input_indices) > 0:
            gpt_input_indices = torch.tensor(gpt_input_indices, dtype=torch.long, device=self.device)
            kv_indices = torch.tensor(kv_indices, dtype=torch.long, device=self.device)
            # gpt_inputs = self.composition_cache[gpt_input_indices[:, 0], gpt_input_indices[:, 1], gpt_input_indices[:, 2], :]
            # gpt_inputs = self.layer_norm(self.gpt_input_cache[gpt_input_indices[:,0], gpt_input_indices[:, 1]])
            gpt_inputs = self.gpt_input_cache[gpt_input_indices[:, 0], gpt_input_indices[:, 1]].unsqueeze(-1) # (?, 1)
            position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
            action_kv = [(k[kv_indices[:, 0], kv_indices[:, 1], :, :, :], v[kv_indices[:, 0], kv_indices[:, 1], :, :, :])
                         for k,v in self.action_key_values]  # (?, num_head, L, head_dim)
            # TODO: add additional mask by current state
            action_masks = self.action_masks[kv_indices[:, 0], kv_indices[:, 1], :]  #(?, L)
            action_masks = action_masks & tg_masks
            # for i in range(min(len(states_batch[0]), 10)):
            #     print("sample: ", i)
            #     print(states_batch[0][i].current_seq)
            #     print(states_batch[0][i].current_step)
            #     print(states_batch[0][i].token_offset)
            #     print(states_batch[0][i].all_closing_nt_cnt)
            #     print(states_batch[0][i].stack)
            #     print(states_batch[0][i].composed_positions)
            #     print(states_batch[0][i].cur_composition_positions)
            #     print(states_batch[0][i].reduce2_pos_seq)
            # breakpoint()             
            # for i in range(len(kv_indices)):
            #     print(states_batch[kv_indices[i,0]][kv_indices[i,1]].current_seq)
            #     print(states_batch[kv_indices[i,0]][kv_indices[i,1]].composed_positions)
            # print(action_masks)
            return gpt_inputs, position_ids, action_kv, action_masks
        else:
            return None, None, None, None