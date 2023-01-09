import requests
from PIL import Image

from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="openai/clip-vit-base-patch32",
    clip_processor_path="openai/clip-vit-base-patch32",
    lang_encoder_path="facebook/opt-125m",
    tokenizer_path="facebook/opt-125m",
)

tokenizer.padding_side = "left"  # we want to pad on the left side for generation

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
vis_x = image_processor(images=[image, image], return_tensors="pt")["pixel_values"]

# add frame and num_images demensions
vis_x = vis_x.unsqueeze(1).unsqueeze(1)

lang_x = tokenizer(
    ["<image>An animal", "<image>A cat<|endofchunk|>"], max_length=10, padding=True, truncation=True, return_tensors="pt"
)

generation = tokenizer.batch_decode(model.generate(vis_x, lang_x["input_ids"], attention_mask=lang_x["attention_mask"], max_length=20))
