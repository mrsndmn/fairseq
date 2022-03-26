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
    --prune-heads-with-hcg \
    --path checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_1.5/checkpoint_last.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 7 --remove-bpe \
    --print-step \
    --batch-size 400


# batch_size = 1200
#
# TRIMMED
#
# 2022-03-26 19:18:01 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,218 tokens) in 29.7s (101.03 sentences/s, 2934.17 tokens/s)
# Generate test with beam=7: BLEU4 = 15.37, 47.8/20.7/10.4/5.7 (BP=0.986, ratio=0.986, syslen=63602, reflen=64506)
#
#
# ORIG
#
# 2022-03-26 19:17:59 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,249 tokens) in 29.4s (102.14 sentences/s, 2967.69 tokens/s)
# Generate test with beam=7: BLEU4 = 15.56, 47.9/20.8/10.5/5.9 (BP=0.987, ratio=0.987, syslen=63688, reflen=64506)
#
#
# batch_size = 400
#
# TRIMMED
#
# 2022-03-26 19:10:14 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,218 tokens) in 24.1s (124.78 sentences/s, 3624.14 tokens/s)
# Generate test with beam=7: BLEU4 = 15.37, 47.8/20.7/10.4/5.7 (BP=0.986, ratio=0.986, syslen=63602, reflen=64506)
#
# ORIG
#
# 2022-03-26 19:12:48 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,249 tokens) in 28.6s (105.14 sentences/s, 3054.71 tokens/s)
# Generate test with beam=7: BLEU4 = 15.56, 47.9/20.8/10.5/5.9 (BP=0.987, ratio=0.987, syslen=63688, reflen=64506)
#
# batch_size = 100
#
# TRIMMED
#
# 2022-03-26 19:15:57 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,218 tokens) in 22.4s (133.85 sentences/s, 3887.52 tokens/s)
# Generate test with beam=7: BLEU4 = 15.37, 47.8/20.7/10.4/5.7 (BP=0.986, ratio=0.986, syslen=63602, reflen=64506)
#
# ORIG
#
# 2022-03-26 19:14:39 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,249 tokens) in 26.8s (111.97 sentences/s, 3253.13 tokens/s)
# Generate test with beam=7: BLEU4 = 15.56, 48.0/20.8/10.5/5.9 (BP=0.987, ratio=0.987, syslen=63688, reflen=64506)
#
#