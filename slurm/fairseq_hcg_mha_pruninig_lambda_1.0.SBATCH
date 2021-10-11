#!/bin/bash
#SBATCH --job-name="hcg_mha_fairseq"
#SBATCH --nodes=1
#SBATCH --time-min=1400
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH -C type_a|type_b|type_c
#SBATCH --mail-user=dyutarasov@edu.hse.ru
#SBATCH --mail-type=ALL

module purge
module load Python/Anaconda_v11.2020

source activate fairseq_hcg_mha

export PYTHONPATH=/home/dyutarasov/workspace/fairseq

set -eo pipefail
set -o xtrace

#Executable
srun nvidia-smi

srun fairseq-train \
    data-bin/wmt14_en_de_distill \
    --save-dir checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_1.0 \
    --with_hard_concrete_gate \
    --hcg_l0_penalty_lambda 1.0 \
    --tensorboard-logdir tensorboard_checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_1.0 \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch nonautoregressive_transformer_hcg \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 18000 \
    --save-interval-updates 25000 \
    --max-update 600000