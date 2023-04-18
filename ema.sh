#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=ema
#SBATCH --account=datanet
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:0
#SBATCH --output=ema_%j.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive

cd /admin/home-irena/open_flamingo/

BETA=0.999
# BETA=0.9999999
echo $BETA

srun --cpu_bind=v --accel-bind=gn python offline_ema.py /fsx/home-anasawadalla/open_flamingo/open_flamingo/train/run1-opt1.3b-vit-L-14-laion2b-8node-mmc4-full-32-thresh --ema_beta ${BETA} --last_ckpt 79 --lm_path facebook/opt-1.3b