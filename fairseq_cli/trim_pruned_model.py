import argparse
import glob
import re
import os.path
import sys

import matplotlib.ticker as ticker
import numpy as np
from tqdm.auto import tqdm

import pickle

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.modules.hcg_attention import MultiHeadHCGAttention

parser = argparse.ArgumentParser(description='Visualize hard concrete hate weights')
parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint directory path')
parser.add_argument('--output', type=str, help='path_to_model_checkpoint')
parser.add_argument('--dry', action='store_true', help='wether save model or not')

args = parser.parse_args()

checkpoint = args.checkpoint

models_ensemble = checkpoint_utils.load_model_ensemble(utils.split_paths( checkpoint ))
model = models_ensemble[0][0]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


before_pruned_params = count_params(model)

print('before prunning:', before_pruned_params)

for module in model.modules():
    if isinstance(module, MultiHeadHCGAttention):
        module.prune()

after_pruning_params = count_params(model)
print('after prunning:', after_pruning_params)
print(f'num of params reduced to { after_pruning_params / before_pruned_params * 100}')


if not args.dry:
    pass
