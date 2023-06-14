#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=15000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export PYTHONPATH="$PYTHONPATH:open_flamingo"
srun --cpu_bind=v --accel-bind=gn python open_flamingo/open_flamingo/train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --dataset_resampled \
    --batch_size_mmc4 32 \
    --batch_size_laion 64 \
    --train_num_samples_mmc4 125000\
    --train_num_samples_laion 250000 \
    --loss_multiplier_laion 0.2 \
    --workers=4 \
    --run_name OpenFlamingo-3B-vitl-mpt1b \
    --num_epochs 480 \
    --warmup_steps  1875 \
    --mmc4_textsim_threshold 0.24 \
    --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
    --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
    --gradient_checkpointing \
    --report_to_wandb \
