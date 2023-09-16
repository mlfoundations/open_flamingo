#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --job-name=openflamingo

<<com
To use FSDP, please make sure to use Pytorch Nightly > 2.0.1!
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export PYTHONPATH="$PYTHONPATH:open_flamingo"
srun --cpu_bind=v --accel-bind=gn python open_flamingo/open_flamingo/train/train.py \
    --lm_path meta-llama/Llama-2-13b \
    --tokenizer_path meta-llama/Llama-2-13b \
    --model_family flamingo \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --batch_size_mmc4 16 \
    --batch_size_laion 32 \
    --fsdp \
    --fsdp_sharding_strategy hybrid \
    --train_num_samples_mmc4 125000\
    --train_num_samples_laion 250000 \
    --loss_multiplier_laion 0.2 \
    --workers=4 \
    --run_name "fsdp" \
    --num_epochs 480 \
    --warmup_steps  0 \
    --mmc4_textsim_threshold 0.0 \
    --laion_shards "/path/to/laion-samples/{000000..000001}.tar" \
    --mmc4_shards "/path/to/mmc4-samples/{000000..000001}.tar" \
    --report_to_wandb
