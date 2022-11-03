import unittest

from src.eval.evaluate import (evaluate_imagenet_zeroshot, evaluate_vqa, evaluate_coco)
from src.open_flamingo.factory import create_model_and_transforms


class TestEvaluations(unittest.TestCase):
    def test_zeroshot_text_vqa(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")
        
        tokenizer.padding_side = "left"  # we want to pad on the left side for generation

        print(evaluate_vqa(model, tokenizer, image_processor, benchmark_name="TextVQA",
                           batch_size=2, num_samples=2, evaluation_stage="test"))

    def test_zeroshot_imagenet(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")
        
        tokenizer.padding_side = "left"  # we want to pad on the left side for generation

        print(evaluate_imagenet_zeroshot(model, tokenizer,
              image_processor, batch_size=2, num_samples_per_class=1, num_classes=2))

    def test_zeroshot_okvqa(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")
        
        tokenizer.padding_side = "left"  # we want to pad on the left side for generation

        print(evaluate_vqa(model, tokenizer, image_processor, benchmark_name="OKVQA",
                           data_dir="/mmfs1/gscratch/efml/anasa2/data/ok-vqa/train",
                           batch_size=2, num_samples=2))

    def test_zeroshot_coco(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")
        
        tokenizer.padding_side = "left"  # we want to pad on the left side for generation

        print(evaluate_coco(model, tokenizer, image_processor,
              data_dir="/data/yfcc-tmp/data/mscoco", batch_size=2, num_samples=2))


if __name__ == '__main__':
    unittest.main()
