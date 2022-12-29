# Downloads CLIP and LM and saves them for offline use

import transformers
from src.open_flamingo.factory import OPTForCausalLMFlamingo

OPT_MODEL_NAME = "facebook/opt-1.3b"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

PATH_TO_LOCAL_CLIP = "clip"
PATH_TO_LOCAL_LM = "opt"

# Download CLIP
clip_model = transformers.CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = transformers.CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# Save CLIP
clip_model.save_pretrained(PATH_TO_LOCAL_CLIP)
clip_processor.save_pretrained(PATH_TO_LOCAL_CLIP)

# Download LM
lm_model = OPTForCausalLMFlamingo.from_pretrained(OPT_MODEL_NAME)
lm_tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-30b")

# Save LM
lm_model.save_pretrained(PATH_TO_LOCAL_LM)
lm_tokenizer.save_pretrained(PATH_TO_LOCAL_LM)

# When running the train script, use the following arguments:
# --vision_encoder_path PATH_TO_LOCAL_CLIP
# --vision_encoder_path PATH_TO_LOCAL_CLIP
# --lm_path PATH_TO_LOCAL_LM
# --tokenizer_path PATH_TO_LOCAL_LM
# --offline