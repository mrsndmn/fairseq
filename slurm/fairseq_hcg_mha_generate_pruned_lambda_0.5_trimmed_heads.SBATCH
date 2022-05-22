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
    --path checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_0.5/checkpoint_last.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 7 --remove-bpe \
    --print-step \
    --batch-size 1200


# batch_size = 1200
#
# TRIMMED
#
# 2022-05-22 12:26:36 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,926 tokens) in 31.2s (96.32 sentences/s, 2820.21 tokens/s)
# Generate test with beam=7: BLEU4 = 16.98, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64616, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:28:18 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,929 tokens) in 34.3s (87.65 sentences/s, 2566.29 tokens/s)
# Generate test with beam=7: BLEU4 = 17.04, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64645, reflen=64506)
#
# batch_size = 400
#
# TRIMMED
#
# 2022-05-22 12:24:11 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,926 tokens) in 24.0s (125.17 sentences/s, 3665.01 tokens/s)
# Generate test with beam=7: BLEU4 = 16.98, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64616, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:22:28 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,929 tokens) in 26.9s (111.51 sentences/s, 3264.95 tokens/s)
# Generate test with beam=7: BLEU4 = 17.04, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64645, reflen=64506)
#
# batch_size = 100
#
# TRIMMED
#
# 2022-05-22 12:18:32 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,926 tokens) in 21.2s (141.52 sentences/s, 4143.65 tokens/s)
# Generate test with beam=7: BLEU4 = 16.98, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64616, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:19:38 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,929 tokens) in 20.7s (145.21 sentences/s, 4251.89 tokens/s)
# Generate test with beam=7: BLEU4 = 17.05, 49.2/22.3/11.6/6.6 (BP=1.000, ratio=1.002, syslen=64645, reflen=64506)
#