echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate openflamingo
which python

LM_PATH="luodian/llama-7b-hf"
LM_TOKENIZER_PATH="luodian/llama-7b-hf"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
CKPT_PATH="openflamingo/OpenFlamingo-9B/checkpoint.pt"
DEVICE="0"

RANDOM_ID=$$
RESULTS_FILE="results_${RANDOM_ID}.json"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --cross_attn_every_n_layers 4 \
    --device $DEVICE \
    --results_file $RESULTS_FILE \
    --eval_winoground


echo "evaluation complete! results written to ${RESULTS_FILE}"
