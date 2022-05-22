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
    --path checkpoints_nonautoregressive_transformer_hcg_pruning_lambda_1.0/checkpoint_last.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 7 --remove-bpe \
    --print-step \
    --batch-size 100


# batch_size = 1200
#
# TRIMMED
#
# 2022-05-22 12:04:04 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,071 tokens) in 24.0s (125.07 sentences/s, 3626.37 tokens/s)
# Generate test with beam=7: BLEU4 = 16.06, 48.7/21.5/11.0/6.2 (BP=0.985, ratio=0.985, syslen=63570, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:06:00 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,068 tokens) in 28.0s (107.40 sentences/s, 3114.06 tokens/s)
# Generate test with beam=7: BLEU4 = 16.08, 48.6/21.5/11.0/6.2 (BP=0.986, ratio=0.986, syslen=63600, reflen=64506)
#
# batch_size = 400
#
# TRIMMED
#
# 2022-05-22 12:08:59 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,071 tokens) in 19.4s (154.84 sentences/s, 4489.51 tokens/s)
# Generate test with beam=7: BLEU4 = 16.06, 48.7/21.5/11.0/6.2 (BP=0.985, ratio=0.985, syslen=63570, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:12:25 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,068 tokens) in 22.1s (136.06 sentences/s, 3944.77 tokens/s)
# Generate test with beam=7: BLEU4 = 16.08, 48.6/21.5/11.0/6.2 (BP=0.986, ratio=0.986, syslen=63600, reflen=64506)
#
# batch_size = 100
#
# TRIMMED
#
# 2022-05-22 12:15:08 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,071 tokens) in 17.5s (171.35 sentences/s, 4968.36 tokens/s)
# Generate test with beam=7: BLEU4 = 16.06, 48.7/21.5/11.0/6.2 (BP=0.985, ratio=0.985, syslen=63570, reflen=64506)
#
# ORIG
#
# 2022-05-22 12:13:43 | INFO | fairseq_cli.generate | Translated 3,003 sentences (87,068 tokens) in 20.1s (149.58 sentences/s, 4337.01 tokens/s)
# Generate test with beam=7: BLEU4 = 16.08, 48.6/21.5/11.0/6.2 (BP=0.986, ratio=0.986, syslen=63600, reflen=64506)
#
#