#!/bin/bash
#SBATCH --job-name="hcg_mha_fairseq"
#SBATCH --nodes=1
#SBATCH --time-min=30
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH -C type_a|type_b|type_c

module purge
module load Python/Anaconda_v11.2020

source activate fairseq_hcg_mha

export PYTHONPATH=/home/dyutarasov/workspace/fairseq

set -eo pipefail
set -o xtrace

#Executable
srun nvidia-smi

srun fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints_nonautoregressive_transformer_orig/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 7 --remove-bpe \
    --print-step \
    --batch-size 400
