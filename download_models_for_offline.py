# Downloads CLIP and LM and saves them for offline use

import transformers
from src.open_flamingo.factory import OPTForCausalLMFlamingo

PATH_TO_LOCAL_CLIP = "clip"
PATH_TO_LOCAL_LM = "opt"

# Download CLIP
clip_model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Save CLIP
clip_model.save_pretrained(PATH_TO_LOCAL_CLIP)
clip_processor.save_pretrained(PATH_TO_LOCAL_CLIP)

# Download LM
lm_model = OPTForCausalLMFlamingo.from_pretrained("facebook/opt-125m")
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