import unittest

import requests
from PIL import Image

from src.open_flamingo.factory import create_model_and_transforms


class TestFlamingoModel(unittest.TestCase):
    def test_forward_pass(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")

        image = Image.open(requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        vis_x = image_processor(images=[image, image], return_tensors="pt")[
            "pixel_values"]
        lang_x = tokenizer(["<image> A dog", "<image> A cat"], max_length=10,
                           padding=True, truncation=True,
                           return_tensors="pt")

        # try batched forward pass
        model(vis_x, lang_x["input_ids"],
              attention_mask=lang_x["attention_mask"])

    def test_generate(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")

        tokenizer.padding_side = "left"  # we want to pad on the left side for generation

        image = Image.open(requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        vis_x = image_processor(images=[image, image], return_tensors="pt")[
            "pixel_values"]
        lang_x = tokenizer(["<image> A dog", "<image> A cat <|endofchunk|>"], max_length=10,
                           padding=True, truncation=True, return_tensors="pt")

        # try batched generation
        out = model.generate(vis_x, lang_x["input_ids"],
                             attention_mask=lang_x["attention_mask"],
                             max_length=20)


if __name__ == '__main__':
    unittest.main()
