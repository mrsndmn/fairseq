import argparse
import glob
import re
import os.path
import sys

import matplotlib.ticker as ticker
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import pickle

from fairseq import checkpoint_utils, options, tasks, utils

parser = argparse.ArgumentParser(description='Visualize hard concrete hate weights')
parser.add_argument('--checkpoints_glob', type=str, help='checkpoint directory path')
parser.add_argument('--ignore_cached', default=False, action="store_true", help='ignore cached p_opened')
parser.add_argument('--only_count_pruned_heads', default=False, action="store_true", help='exit after counting pruned heads')


args = parser.parse_args()


checkpoints = list(glob.glob(args.checkpoints_glob))

checkpoints_by_num = []
for checkpoint in checkpoints:
    matches = re.findall(r'checkpoint(\d+).pt$', checkpoint)
    if len(matches) ==0 :
        # print("invalid checkpoint:", checkpoint)
        continue

    checkpoints_num = int(matches[0])
    checkpoints_by_num.append( (checkpoint, checkpoints_num) )

checkpoints_by_num = sorted(checkpoints_by_num, key=lambda x: x[1])

checkpoints_sorted = [ x[0] for x in checkpoints_by_num ]

encoder_attention_p_open = []
decoder_self_attention_p_open = []
decoder_encoder_attention_p_open = []

for checkpoint in tqdm(checkpoints_sorted, desc="reading_checkpoints"):

    encoder_self_attention_p_open_by_layer = []
    decoder_self_attention_p_open_by_layer = []
    decoder_encoder_attention_p_open_by_layer = []


    checkpoint_p_opened = checkpoint + ".mhas_probas.pickle"
    if not args.ignore_cached and os.path.exists(checkpoint_p_opened):
        with open(checkpoint_p_opened, "rb") as f:
            encoder_self_attention_p_open_by_layer, decoder_self_attention_p_open_by_layer, decoder_encoder_attention_p_open_by_layer = pickle.load(f)
    else:
        models_ensemble = checkpoint_utils.load_model_ensemble(utils.split_paths( checkpoint ))
        model = models_ensemble[0][0]

        for i, l in enumerate(model.encoder.layers):
            l.self_attn.hcg.eval()
            if not l.self_attn.with_hard_concrete_gate:
                print("encoder with_hard_concrete_gate")
            encoder_self_attention_p_open_by_layer.append(l.self_attn.hcg.get_p_open().tolist())

        for i, l in enumerate(model.decoder.layers):
            l.self_attn.hcg.eval()
            if not l.self_attn.with_hard_concrete_gate:
                print("decoder self_attn with_hard_concrete_gate")

            decoder_self_attention_p_open_by_layer.append(l.self_attn.hcg.get_p_open().tolist())

        for i, l in enumerate(model.decoder.layers):
            l.encoder_attn.hcg.eval()
            if not l.encoder_attn.with_hard_concrete_gate:
                print("decoder encoder_attn with_hard_concrete_gate")
            decoder_encoder_attention_p_open_by_layer.append(l.encoder_attn.hcg.get_p_open().tolist())

        with open(checkpoint_p_opened, "wb") as f:
            pickle.dump([encoder_self_attention_p_open_by_layer, decoder_self_attention_p_open_by_layer, decoder_encoder_attention_p_open_by_layer], f)

    encoder_attention_p_open.append(encoder_self_attention_p_open_by_layer)
    decoder_self_attention_p_open.append(decoder_self_attention_p_open_by_layer)
    decoder_encoder_attention_p_open.append(decoder_encoder_attention_p_open_by_layer)


def count_pruned_heads(p_opens, threshold=0.9):
    p_opens = np.array(p_opens)
    p_opens = p_opens[-1, ...]
    return (p_opens < threshold).sum()

def count_heads(p_opens):
    p_opens = np.array(p_opens)
    p_opens = p_opens[-1, ...]
    return p_opens.size

print("count_pruned_heads: encoder_attention         =", count_pruned_heads(encoder_attention_p_open))
print("       total count: encoder_attention         =", count_heads(encoder_attention_p_open))
print("count_pruned_heads: decoder_self_attention    =", count_pruned_heads(decoder_self_attention_p_open))
print("       total count: decoder_self_attention    =", count_heads(decoder_self_attention_p_open))
print("count_pruned_heads: decoder_encoder_attention =", count_pruned_heads(decoder_encoder_attention_p_open))
print("       total count: decoder_encoder_attention =", count_heads(decoder_encoder_attention_p_open))

if args.only_count_pruned_heads:
    sys.exit(0)

def draw_p_opens(p_opens_by_layer):
    for layer_p_opens in p_opens_by_layer:
        pass


def draw_mha_p_opened(mha_p_opened_by_checkpoint, file_name):
    layer_p_open = mha_p_opened_by_checkpoint[0]
    first_layer = layer_p_open[0]
    for layer in layer_p_open:
        # print("layer", layer)
        assert len(first_layer) == len(layer)

    num_layers = len(layer_p_open)
    num_heads = len(first_layer)


    fig, ax = plt.subplots()

    step_pixels = 64

    def init():
        ax.set_xlim(0, num_heads * step_pixels)
        ax.set_ylim(0, num_layers * step_pixels)

        plt.xticks(np.arange(1, num_heads*step_pixels, step_pixels) + int(step_pixels / 2),  np.arange(1, num_heads+1, 1))
        plt.yticks(np.arange(1, num_layers*step_pixels, step_pixels) + int(step_pixels / 2), np.arange(1, num_layers+1, 1))

        ax.set_ylabel("layers")
        ax.set_xlabel("heads")

        return []

    def update(layer_p_open):

        patches=[]
        for i, layer in enumerate(layer_p_open):
            for j, head in enumerate(layer):
                patches.append( ax.add_patch( Rectangle(( j * step_pixels , i * step_pixels), step_pixels, step_pixels, color="white") ) )
                patches.append( ax.add_patch( Rectangle(( j * step_pixels , i * step_pixels), step_pixels, step_pixels, color="black", alpha=1 - head) ) )

        return patches

    anim = FuncAnimation(fig, update, frames=mha_p_opened_by_checkpoint,
                        init_func=init, blit=True)
    writergif = animation.PillowWriter(fps=10)
    anim.save(file_name, writer=writergif)

checkpoints_dir = os.path.dirname(args.checkpoints_glob)


print("plotting encoder_attention_p_open")
draw_mha_p_opened(encoder_attention_p_open, checkpoints_dir + "/encoder_attention_p_open.gif")

print("plotting decoder_self_attention_p_open")
draw_mha_p_opened(decoder_self_attention_p_open, checkpoints_dir + "/decoder_self_attention_p_open.gif")

print("plotting decoder_encoder_attention_p_open")
draw_mha_p_opened(decoder_encoder_attention_p_open, checkpoints_dir + "/decoder_encoder_attention_p_open.gif")

print(checkpoints_dir + "/encoder_attention_p_open.gif")
print(checkpoints_dir + "/decoder_self_attention_p_open.gif")
print(checkpoints_dir + "/decoder_encoder_attention_p_open.gif")

