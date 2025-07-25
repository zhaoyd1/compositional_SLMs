import argparse
import json
import logging
import os
import random
import re
import sys
import tqdm
import math

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from eval.evaluate_lm import (R2D2GenEvaluator, R2D2GenFastEvaluator,
                              VanillaGPTEvaluator)
from utils.beam_searcher_0100 import R2D2GenFastBeamSearcher
from model.model_factory import create_model
from utils.vocab_builder import load_dict
from tokenizers import AddedToken

test_suite_dict = {}
test_suite_dict["Agreement"] = ["number_orc", "number_prep", "number_src"]
test_suite_dict["Center_Embedding"] = ["center_embed", "center_embed_mod"]
test_suite_dict["Garden_Path_Effects"] = ["mvrr", "mvrr_mod", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod"]
test_suite_dict["Cross_Syntactic_Expectation"] = ["subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"]
test_suite_dict["Licensing"] = ["npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever", \
    "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc"]
test_suite_dict["Long_Distance_Dependencies"] = ["fgd_subject", "fgd_object", "fgd_pp", "fgd_embed3", "fgd_embed4", "fgd_hierarchy", "cleft", "cleft_mod"]


task_list = ["center_embed", "center_embed_mod", "cleft", "cleft_mod", "fgd_subject", "fgd_object", "fgd_pp", "fgd_embed3", \
    "fgd_embed4", "fgd_hierarchy", "mvrr", "mvrr_mod", "nn_nv_rpl", "npi_orc_any", "npi_orc_ever", "npi_src_any", \
    "npi_src_ever", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "number_orc", "number_prep", "number_src", \
    "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc", \
    "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"]


formula_dict = {
    'center_embed': ['[ (6;%plaus%) + (7;%plaus%) ] < [ (6;%implaus%) + (7;%implaus%) ]'],
    'center_embed_mod': ['[ (7;%plaus%) + (8;%plaus%) ] < [ (7;%implaus%) + (8;%implaus%) ]'],
    'cleft': ['[(6;%np_mismatch%)-(6;%np_match%)]+[[(5;%vp_mismatch%)+(6;%vp_mismatch%)]-[(5;%vp_match%)+(6;%vp_match%)]]>0'],
    'cleft_mod': ['[(7;%np_mismatch%)-(7;%np_match%)]+[[(6;%vp_mismatch%)+(7;%vp_mismatch%)]-[(6;%vp_match%)+(7;%vp_match%)]]>0'],
    'fgd_subject': ['[(3;%what_nogap%) > (3;%that_nogap%) ] & [(4;%what_gap%) < (4;%that_gap%) ] '],
    'fgd_object': ['[(5;%what_nogap%) > (5;%that_nogap%) ] & [(6;%what_gap%) < (6;%that_gap%) ] '],
    'fgd_pp': ['[(7;%what_nogap%) > (7;%that_nogap%) ] & [(8;%what_gap%) < (8;%that_gap%) ] '],
    'fgd_embed3': ['[(6;%what_no-gap%)>(6;%that_no-gap%)] & [(7;%what_gap%)<(7;%that_gap%)]'],
    'fgd_embed4': ['[(6;%what_no-gap%)>(6;%that_no-gap%)] & [(7;%what_gap%)<(7;%that_gap%)]'],
    'fgd_hierarchy': ['[(6;%what_nogap%) >  (6;%that_nogap%)] & [(6;%what_subjgap%) <  (6;%that_subjgap%)]','[(9;%what_nogap%) =  (9;%that_nogap%)]& [(6;%what_subjgap%) =  (6;%that_subjgap%)]'],
    'mvrr': ['[(5;%reduced_ambig%) > (5;%unreduced_ambig%)] & [(5;%reduced_ambig%) > (5;%reduced_unambig%)] & [[(5;%reduced_ambig%) - (5;%unreduced_ambig%)] > [(5;%reduced_unambig%) - (5;%unreduced_unambig%)]]'],
    'mvrr_mod': ['[(6;%reduced_ambig%) > (6;%unreduced_ambig%)] & [(6;%reduced_ambig%) > (6;%reduced_unambig%)] & [[(6;%reduced_ambig%) - (6;%unreduced_ambig%)] > [(6;%reduced_unambig%) - (6;%unreduced_unambig%)]]'],
    'nn_nv_rpl': ['(5;%nn_ambig%)>(5;%nn_unambig%)', '(5;%nv_ambig%)>(5;%nv_unambig%)'], 
    'npi_orc_any': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_orc_ever': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_src_any': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_src_ever': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'], 
    'npz_ambig': ['[(5;%ambig_nocomma%) > (5;%ambig_comma%) ] &  [(5;%ambig_nocomma%) > (5;%unambig_nocomma%) ]  & [[(5;%ambig_nocomma%) - (5;%ambig_comma%) ] > [(5;%unambig_nocomma%) - (5;%unambig_comma%) ]]'],
    'npz_ambig_mod': ['[(6;%ambig_nocomma%) > (6;%ambig_comma%) ] &  [(6;%ambig_nocomma%) > (6;%unambig_nocomma%) ]  & [[(6;%ambig_nocomma%) - (6;%ambig_comma%) ] > [(6;%unambig_nocomma%) - (6;%unambig_comma%) ]]'],
    'npz_obj': ['[(5;%no-obj_no-comma%) > (5;%no-obj_comma%) ] &  [(5;%no-obj_no-comma%) > (5;%obj_no-comma%) ] & [[(5;%no-obj_no-comma%) - (5;%no-obj_comma%) ] > [(5;%obj_no-comma%) - (5;%obj_comma%) ]]'],
    'npz_obj_mod': ['[(6;%no-obj_no-comma%) > (6;%no-obj_comma%) ] &  [(6;%no-obj_no-comma%) > (6;%obj_no-comma%) ] & [[(6;%no-obj_no-comma%) - (6;%no-obj_comma%) ] > [(6;%obj_no-comma%) - (6;%obj_comma%) ]]'],
    'number_orc': ['[(7;%match_sing%) < (7;%mismatch_sing%)] & [(7;%match_plural%) < (7;%mismatch_plural%)]'],
    'number_prep': ['[(6;%match_sing%) < (6;%mismatch_sing%)] & [(6;%match_plural%) < (6;%mismatch_plural%)]'],
    'number_src': ['[(7;%match_sing%) < (7;%mismatch_sing%)] & [(7;%match_plural%) < (7;%mismatch_plural%)]'],
    'reflexive_orc_fem': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (7;%match_sing%) < (7;%mismatch_sing%) ] & [ (7;%match_plural%) < (7;%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (7;%match_sing%) < (7;%mismatch_sing%) ] & [ (7;%match_plural%) < (7;%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'subordination': ['[(3;%sub_no-matrix%) > (3;%no-sub_no-matrix%) ] & [(3;%sub_matrix%) < (3;%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]']
}


new_formula_dict = {
    'center_embed': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    'center_embed_mod': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    
    'cleft': ['[ (%np_mismatch%) - (%np_match%) ] + [ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    'cleft_mod': ['[ (%np_mismatch%) - (%np_match%) ]+[ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    
    'fgd_subject': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_object': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_pp': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_embed3': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_embed4': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_hierarchy': ['[ (%what_nogap%) > (%that_nogap%)] & [ (%what_subjgap%) <  (%that_subjgap%) ]', '[ (%what_nogap%) = (%that_nogap%) ] & [ (%what_subjgap%) = (%that_subjgap%) ]'],
    
    'mvrr': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%) ] > [ (%reduced_unambig%) - (%unreduced_unambig%) ] ]'],
    'mvrr_mod': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%)] > [(%reduced_unambig%) - (%unreduced_unambig%)] ]'],
    
    'nn_nv_rpl': ['(%nn_ambig%)>(%nn_unambig%)', '(%nv_ambig%)>(%nv_unambig%)'], 
    
    'npi_orc_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_orc_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'], 
    
    'npz_ambig': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_ambig_mod': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_obj': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    'npz_obj_mod': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    
    'number_orc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_prep': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_src': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'reflexive_orc_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'subordination': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]']
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _scalar(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    else:
        raise RuntimeError("Not value tensor")

class Scorer():
    # load an input and get score
    def __init__(self, model_type, model, config, device, beam_size=300, ext_vocab=None):
        self.model_type = model_type
        self.model = model
        self.device = device
        if model_type == "gpt":
            self.core = VanillaGPTEvaluator(self.model)
        elif model_type == "r2d2-gen":
            self.core = R2D2GenEvaluator(self.model, config, device, beam_size=beam_size, ext_vocab=ext_vocab)
        elif model_type == "tg":
            self.core = R2D2GenFastBeamSearcher(self.model, config, device, beam_size=beam_size, sampling=False)
        else:
            raise RuntimeError("Invalid model type.")
    
    def score(self, input_ids, tags):
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        tags = torch.tensor(tags, dtype=torch.long).to(self.device)
        if self.model_type == "r2d2-gee":
            with torch.no_grad():
                result = self.core.beam_search(input_ids.unsqueeze(0), tag=tags.unsqueeze(0))[0]
        elif self.model_type == "tg":
            with torch.no_grad():
                result = self.core.beam_search(target_ids=input_ids.unsqueeze(0), target_masks=torch.ones_like(input_ids.unsqueeze(0)), tag=tags.unsqueeze(0))[0]
        elif self.model_type == "gpt":
            with torch.no_grad():
                result = self.core.perplexity(input_ids, tags)[0]
        return result


class Evaluator():
    # load the whole dataset of "task_name" and compute accuracy
    def __init__(self, task_path, model_type, model, tokenizer, config, device, beam_size=300, ext_vocab=None):
        self.task_path = task_path
        task_name = task_path.split('/')[-1][:-5]
        self.task_name = task_name
        assert self.task_name in task_list
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.gpt_config = config
        self.device = device
        self.beam_size = beam_size
        self.ext_vocab = ext_vocab
        self.loaddata()
        self.scorer = Scorer(self.model_type, self.model, self.gpt_config, self.device, beam_size=self.beam_size, ext_vocab=self.ext_vocab)
    
    def loaddata(self):
        with open(self.task_path, 'r') as file:
            data = json.load(file)
        self.data = data
    
    def eval_math_expr(self, expr):
        try:
            return eval(expr)
        except:
            return math.nan
    
    def run(self, formula, input_list):
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        score_dict = {}
        for item in input_list:
            # add a space before input
            input_ids = self.tokenizer.encode(item["input"])
            tag = item["tag"][0]
            assert len(input_ids) == len(tag), "tags should have same length with tokenized input_ids"
            score_dict[item["condition_name"]] = _scalar(self.scorer.score(input_ids, tag))
            print("input: ", item["input"], "condition_name: ",item["condition_name"], "score: ", score_dict[item["condition_name"]])
        
        for key in keys:
            formula = formula.replace(
                "(%{}%)".format(key),
                str(score_dict[key]),
                )
        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        # print(formula)
        return self.eval_math_expr(formula)
    
    def eval(self):
        # TODO: write new formula into processed json, deal with multiple formulas in one task
        formula = new_formula_dict[self.task_name][0]
        total_len = len(self.data["data"])
        po = 0
        for data in self.data["data"]:
            # print("data: ", data)
            po += self.run(formula, data)
        acc = po / total_len
        return acc


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments for sg evaluator")
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--ext_vocab_path', required=False, default=None, type=str, help='external vocab path')
    cmd.add_argument('--task_name', default="cleft", type=str, help='task name of sg')
    cmd.add_argument('--task_path', required=False, type=str, help="path to the sg task data")
    cmd.add_argument('--task_dir', required=False, type=str, help="directory of the sg task data")
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2', 'r2d2-gen-fast', 'r2d2-fast', 'tg', 'n-ary-gen'], default='r2d2-gen-fast')
    # cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--seed', type=int, default=3407)
    cmd.add_argument('--fix_embedding', action='store_true')
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--beam_size', default=300, type=int)
    cmd.add_argument('--gradient_checkpoint', action='store_true')
    cmd.add_argument('--alltest', action='store_true')
    cmd.add_argument('--tg_enabled', required=False, default=0, type=int)
    cmd.add_argument('--output_path', required=True, default='sg_result/test.txt', type=str)
    cmd.add_argument('--cuda_num', required=False, default=0, type=int)

    args = cmd.parse_args(sys.argv[1:])

    model = create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path, args.fix_embedding, args.gradient_checkpoint)
    print("model initialized")
    
    if args.pretrain_dir is not None:
        if args.model_type == 'tg':
            state_dicts = torch.load(os.path.join(args.pretrain_dir, 'model.bin'), map_location=lambda a, b: a)
            out_dict = {}
            for key, val in state_dicts.items():
                if 'module.gpt.' in key:
                    new_key = key.replace('module.gpt.', 'gpt.')
                elif 'module.' in key:
                    new_key = key.replace('module.', '')
                out_dict[new_key] = val
            model.load_state_dict(out_dict)
        else:
            model.from_pretrain(args.pretrain_dir, strict=True)
    print("load from pretrain dir successfully")
    
    device = torch.device(f'cuda:{args.cuda_num}')
    set_seed(args.seed)
    model.to(device=device)

    score_dict = {}
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    if args.tg_enabled == 1:
        non_terminal_token = ['(ADJP', '(ADVP', '(CONJP', '(FRAG', '(INTJ', '(LST', '(NAC', '(NP', '(NX', '(PP', '(PRN', '(PRT', '(QP', '(RRC', '(S', '(SBAR', '(SBARQ', '(SINV', '(SQ', '(UCP', '(VP', '(WHADJP', '(WHADVP', '(WHNP', '(WHPP', '(X', \
                              'ADJP)', 'ADVP)', 'CONJP)', 'FRAG)', 'INTJ)', 'LST)', 'NAC)', 'NP)', 'NX)', 'PP)', 'PRN)', 'PRT)', 'QP)', 'RRC)', 'S)', 'SBAR)', 'SBARQ)', 'SINV)', 'SQ)', 'UCP)', 'VP)', 'WHADJP)', 'WHADVP)', 'WHNP)', 'WHPP)', 'X)']
        tokens = [AddedToken(token, lstrip=True) for token in non_terminal_token]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    config = AutoConfig.from_pretrained(args.gpt_config_path)
    if args.ext_vocab_path:
        ext = load_dict(args.ext_vocab_path)
    else: 
        ext = None
    if args.alltest:
        output_path = args.output_path
        output = open(output_path, 'w')
        sumscore = 0
        averagescore = 0
        for taskname in task_list:
            # if taskname != "npi_orc_ever":
            #     continue
            # taskpath = "/".join(args.task_path.split('/')[:-1]) + "/" + taskname + ".json"
            taskpath = os.path.join(args.task_dir, f"{taskname}.json")
            evaluator = Evaluator(taskpath, args.model_type, model, tokenizer, config, device, beam_size=args.beam_size, ext_vocab=ext)
            acc = evaluator.eval()
            score_dict[taskname] = acc
            print("----------------------------------------------------task_name: ", taskname, "performance: ", acc, "----------------------------------------------------")
            output.write("task_name: " + taskname + " performance: " + str(acc) + "\n")
        print("----------------------------------------------score_dict-----------------------------------------------------", score_dict)
        final_score_dict = {}
        for item in test_suite_dict.keys():
            tempsum = 0
            templen = len(test_suite_dict[item])
            for name in test_suite_dict[item]:
                tempsum += score_dict[name]
            final_score_dict[item] = tempsum / templen
        for item in final_score_dict.keys():
            sumscore += final_score_dict[item]
        averagescore = sumscore / 6
        final_score_dict["average"] = averagescore
        print(final_score_dict)
        output.write(json.dumps(final_score_dict, indent=4) + "\n")
    else:
        evaluator = Evaluator(args.task_path, args.model_type, model, tokenizer, config, device, beam_size=args.beam_size, ext_vocab=ext)
        acc = evaluator.eval()
        print("task_name: ", args.task_name, "performance: ", acc)