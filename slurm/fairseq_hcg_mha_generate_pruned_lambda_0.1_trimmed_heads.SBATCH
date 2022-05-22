#!/bin/bash
#SBATCH --job-name="1.5_gen"
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
    --batch-size 1200


# batch_size = 1200
#
# TRIMMED
#
# 2022-05-22 12:26:39 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 33.2s (90.59 sentences/s, 2653.51 tokens/s)
# Generate test with beam=7: BLEU4 = 17.17, 49.4/22.4/11.7/6.7 (BP=0.999, ratio=0.999, syslen=64411, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:28:19 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 34.9s (86.12 sentences/s, 2522.66 tokens/s)
# Generate test with beam=7: BLEU4 = 17.20, 49.4/22.4/11.8/6.7 (BP=0.999, ratio=0.999, syslen=64418, reflen=64506)
#
# batch_size = 400
#
# TRIMMED
#
# 2022-05-22 12:24:16 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 25.3s (118.88 sentenc
# Generate test with beam=7: BLEU4 = 17.17, 49.4/22.4/11.7/6.7 (BP=0.999, ratio=0.999, syslen=64411, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:22:29 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 27.3s (109.86 sentences/s, 3218.10 tokens/s)
# Generate test with beam=7: BLEU4 = 17.20, 49.4/22.4/11.8/6.7 (BP=0.999, ratio=0.999, syslen=64418, reflen=64506)
#
# batch_size = 100
#
# TRIMMED
#
# 2022-05-22 12:18:41 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 21.9s (137.09 sentences/s, 4015.54 tokens/s)
# Generate test with beam=7: BLEU4 = 17.17, 49.4/22.4/11.7/6.7 (BP=0.999, ratio=0.999, syslen=64411, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:20:04 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,964 tokens) in 20.6s (145.86 sentenc
# Generate test with beam=7: BLEU4 = 17.20, 49.4/22.4/11.8/6.7 (BP=0.999, ratio=0.999, syslen=64418, reflen=64506)
#