#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=6
#SBATCH --gpus-per-task=1
#SBATCH --account=efml
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=flamingo

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=15000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export HF_DATASETS_CACHE="/gscratch/efml/anasa2/.huggingface" TRANSFORMERS_CACHE="/gscratch/efml/anasa2/.huggingface"

export PYTHONPATH="$PYTHONPATH:open_flamingo"
srun --cpu_bind=v --accel-bind=gn python 



deepspeed open_flamingo/open_flamingo/train/train.py \
    --lm_path meta-llama/Llama-2-13b \
    --tokenizer_path meta-llama/Llama-2-13b \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --batch_size_mmc4 16 \
    --batch_size_laion 32 \
    --deepspeed \
    --train_num_samples_mmc4 125000\
    --train_num_samples_laion 250000 \
    --loss_multiplier_laion 0.2 \
    --workers=4 \
    --run_name "deepspeed" \
    --num_epochs 480 \
    --warmup_steps  0 \
    --mmc4_textsim_threshold 0.0 \
    --laion_shards "/mmfs1/gscratch/efml/anasa2/laion-samples/{000000..000001}.tar" \
    --mmc4_shards "/mmfs1/gscratch/efml/anasa2/mmc4-samples/shard_{0..1}-000000000.tar" \
    --gradient_checkpointing \
    --report_to_wandb \
