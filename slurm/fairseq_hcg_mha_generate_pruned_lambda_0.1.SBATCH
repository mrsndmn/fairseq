#!/bin/bash
#SBATCH --job-name="generate_pruned_0.1"
#SBATCH --nodes=1
#SBATCH --time-min=30
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH -C type_a|type_b|type_c

module purge
module load Python/Anaconda_v11.2020

source activate fairseq_hcg_mha

export PYTHONPATH=/home/dyutarasov/workspace/fairseq

set -eo pipefail
set -o xtrace

#Executable
# srun nvidia-smi

srun fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_0.1/checkpoint_last.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 7 --remove-bpe \
    --print-step \
    --batch-size 400
