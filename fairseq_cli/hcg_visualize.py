import argparse
import glob
import re
import os.path

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from fairseq import checkpoint_utils, options, tasks, utils

parser = argparse.ArgumentParser(description='Visualize hard concrete hate weights')
parser.add_argument('--checkpoints_glob', type=str, help='checkpoint directory path')

args = parser.parse_args()


checkpoints = list(glob.glob(args.checkpoints_glob))

checkpoints_by_num = []
for checkpoint in checkpoints:
    matches = re.findall(r'checkpoint(\d+).pt', checkpoint)
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
    models_ensemble = checkpoint_utils.load_model_ensemble(utils.split_paths( checkpoint ))
    model = models_ensemble[0][0]

    encoder_self_attention_p_open_by_layer = []
    for i, l in enumerate(model.encoder.layers):
        l.self_attn.hcg.eval()
        encoder_self_attention_p_open_by_layer.append(l.self_attn.hcg.get_p_open())

    encoder_attention_p_open.append(encoder_self_attention_p_open_by_layer)


    decoder_self_attention_p_open_by_layer = []
    for i, l in enumerate(model.decoder.layers):
        l.self_attn.hcg.eval()
        decoder_self_attention_p_open_by_layer.append(l.self_attn.hcg.get_p_open())

    decoder_self_attention_p_open.append(decoder_self_attention_p_open_by_layer)

    for i, l in enumerate(model.decoder.layers):
        l.encoder_attn.hcg.eval()
        decoder_encoder_attention_p_open.append(l.encoder_attn.hcg.get_p_open())

def draw_p_opens(p_opens_by_layer):
    for layer_p_opens in p_opens_by_layer:
        pass


def draw_mha_p_opened(mha_p_opened_by_checkpoint, file_name):
    layer_p_open = mha_p_opened_by_checkpoint[0]
    first_layer = layer_p_open[0]
    for layer in layer_p_open:
        assert first_layer.shape == layer.shape

    num_layers = len(layer_p_open)
    num_heads = len(first_layer)


    fig, ax = plt.subplots()

    step_pixels = 64

    def init():
        ax.set_xlim(0, num_heads * step_pixels)
        ax.set_ylim(0, num_layers * step_pixels)
        return fig

    def update(layer_p_open):
        fig.clear()

        for i, layer in enumerate(layer_p_open):
            for j, head in enumerate(layer):
                ax.add_patch( Rectangle(( j * step_pixels , i * step_pixels), step_pixels, step_pixels, color="black", alpha=1 - head) )

        return plt.plot()

    anim = FuncAnimation(fig, update, frames=encoder_attention_p_open,
                        init_func=init, blit=True)
    writergif = animation.PillowWriter(fps=30)
    anim.save(file_name, writer=writergif)

checkpoints_dir = os.path.dirname(args.checkpoints_glob)

print("plotting encoder_attention_p_open")
draw_mha_p_opened(encoder_attention_p_open, checkpoints_dir + "/encoder_attention_p_open.gif")

print("plotting encoder_attention_p_open")
draw_mha_p_opened(decoder_self_attention_p_open, checkpoints_dir + "/decoder_self_attention_p_open.gif")

print("plotting encoder_attention_p_open")
draw_mha_p_opened(decoder_encoder_attention_p_open, checkpoints_dir + "/decoder_encoder_attention_p_open.gif")

