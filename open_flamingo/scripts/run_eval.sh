#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

EXP_PATH="/mmfs1/home/irenagao/gscratch/mpt7b-4xattn-0.24thres-8node-ddp"
EPOCH=479
LM="mosaicml/mpt-7b"
TOK=$LM
N=4

SHOTS=32
DATASET="COCO"
# DATASET="VQA"
# DATASET="Flickr"
# DATASET="OKVQA"

if [ $SHOTS == 0 ]; then
    BATCH_SIZE=24
elif [ $SHOTS <= 4 ]; then
    BATCH_SIZE=20
elif [ $SHOTS <= 8 ]; then
    BATCH_SIZE=14
elif [ $SHOTS <= 16 ]; then
    BATCH_SIZE=12
else
    BATCH_SIZE=8
fi

if [ $LM == "mosaicml/mpt-7b" ]; then
    # subtract 4 from batch size for 7b
    BATCH_SIZE=$((BATCH_SIZE-6))
fi

##############################################

# module load openmpi
# source /opt/intel/mpi/latest/env/vars.sh


export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /mmfs1/home/irenagao/gscratch/open_flamingo/open_flamingo/eval
export PYTHONPATH="$PYTHONPATH:/mmfs1/home/irenagao/gscratch/open_flamingo/"

echo $EXP_PATH
echo $EPOCH
echo $SHOTS
echo $DATASET

nvidia-smi

if [ $DATASET == "COCO" ]; then
    srun --cpu_bind=v --accel-bind=gn python evaluate.py \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai\
        --lm_path ${LM} \
        --lm_tokenizer_path ${TOK} \
        --cross_attn_every_n_layers ${N} \
        --shots "${SHOTS}" \
        --num_trials 3 \
        --trial_seeds 0 1 2 \
        --checkpoint_path "${EXP_PATH}/checkpoint_${EPOCH}.pt" \
        --results_file "${EXP_PATH}/checkpoint_${EPOCH}_${DATASET}_${SHOTS}_eval.json" \
        --eval_coco \
        --batch_size ${BATCH_SIZE} \
        --coco_train_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/mscoco_karpathy/train2014" \
        --coco_val_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/mscoco_karpathy/val2014" \
        --coco_karpathy_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/mscoco_karpathy/dataset_coco.json" \
        --coco_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/mscoco_karpathy/annotations/captions_val2014.json" \
        --precision amp_bf16 \

elif [ $DATASET == "VQA" ]; then
    srun --cpu_bind=v --accel-bind=gn python evaluate.py \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai\
        --lm_path ${LM} \
        --lm_tokenizer_path ${TOK} \
        --cross_attn_every_n_layers ${N} \
        --shots "${SHOTS}" \
        --num_trials 3 \
        --trial_seeds 0 1 2 \
        --checkpoint_path "${EXP_PATH}/checkpoint_${EPOCH}.pt" \
        --results_file "${EXP_PATH}/checkpoint_${EPOCH}_${DATASET}_${SHOTS}_eval.json" \
        --eval_vqav2 \
        --batch_size ${BATCH_SIZE} \
        --vqav2_train_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/train2014" \
        --vqav2_train_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/v2_mscoco_train2014_annotations.json" \
        --vqav2_train_questions_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
        --vqav2_test_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/val2014" \
        --vqav2_test_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/v2_mscoco_val2014_annotations.json" \
        --vqav2_test_questions_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json" \
        --precision amp_bf16 \

elif [ $DATASET == "Flickr" ]; then
    # num_samples=-1 for whole dataset
    srun --cpu_bind=v --accel-bind=gn python evaluate.py \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai\
        --lm_path ${LM} \
        --lm_tokenizer_path ${TOK} \
        --cross_attn_every_n_layers ${N} \
        --shots "${SHOTS}" \
        --num_trials 3 \
        --trial_seeds 0 1 2 \
        --checkpoint_path "${EXP_PATH}/checkpoint_${EPOCH}.pt" \
        --results_file "${EXP_PATH}/checkpoint_${EPOCH}_${DATASET}_${SHOTS}_eval.json" \
        --eval_flickr30 \
        --num_samples -1 \
        --batch_size ${BATCH_SIZE} \
        --flickr_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/flickr30k/flickr30k-images" \
        --flickr_karpathy_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/flickr30k/dataset_flickr30k.json" \
        --flickr_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/flickr30k/dataset_flickr30k_coco_style.json" \
        --precision amp_bf16 \

elif [ $DATASET == "OKVQA" ]; then
    # num_samples=-1 for whole dataset
    srun --cpu_bind=v --accel-bind=gn python evaluate.py \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai\
        --lm_path ${LM} \
        --lm_tokenizer_path ${TOK} \
        --cross_attn_every_n_layers ${N} \
        --shots "${SHOTS}" \
        --num_trials 3 \
        --trial_seeds 0 1 2 \
        --checkpoint_path "${EXP_PATH}/checkpoint_${EPOCH}.pt" \
        --results_file "${EXP_PATH}/checkpoint_${EPOCH}_${DATASET}_${SHOTS}_eval.json" \
        --eval_ok_vqa \
        --batch_size ${BATCH_SIZE} \
        --ok_vqa_train_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/train2014" \
        --ok_vqa_train_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/mscoco_train2014_annotations.json" \
        --ok_vqa_train_questions_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
        --ok_vqa_test_image_dir_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/val2014" \
        --ok_vqa_test_annotations_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/mscoco_val2014_annotations.json" \
        --ok_vqa_test_questions_json_path "/mmfs1/gscratch/efml/anasa2/eval_data/okvqa/OpenEnded_mscoco_val2014_questions.json" \
        --precision amp_bf16 \

else
    echo "Invalid dataset"
fi

echo $EXP_PATH
echo $EPOCH
echo $SHOTS
echo $DATASET