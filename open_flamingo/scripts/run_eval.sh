echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate openflamingo
which python

LM_PATH="facebook/opt-1.3b"
LM_TOKENIZER_PATH="facebook/opt-1.3b"
CLIP_PATH="openai/clip-vit-large-patch14"
CKPT_PATH="/mmfs1/gscratch/efml/anasa2/latest_checkpoint.pt"
DEVICE="0"

# Default values for running on klone/hyak; change to override.
COCO_IMG_PATH="/data/yfcc-tmp/data/mscoco/train2017/"
COCO_ANNO_PATH="/data/yfcc-tmp/data/mscoco/annotations/captions_train2017.json"
VQAV2_IMG_PATH="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014"
VQAV2_ANNO_PATH="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json"
VQAV2_QUESTION_PATH="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"

RANDOM_ID=$$
RESULTS_FILE="results_${RANDOM_ID}.json"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --clip_path $CLIP_PATH \
    --checkpoint_path $CKPT_PATH \
    --device $DEVICE \
    --coco_image_dir_path $COCO_IMG_PATH \
    --coco_annotations_json_path $COCO_ANNO_PATH \
    --vqav2_image_dir_path $VQAV2_IMG_PATH \
    --vqav2_annotations_json_path $VQAV2_ANNO_PATH \
    --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
    --eval_coco \
    --eval_vqav2 \
    --results_file $RESULTS_FILE
    # use the below flags to run faster eval during development.
    # --num_samples 16 --shots 0 --num_trials 1


echo "evaluation complete! results written to ${RESULTS_FILE}"
