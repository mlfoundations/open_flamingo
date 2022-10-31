import unittest

from src.eval.evaluate import evaluate_text_vqa, evaluate_imagenet_zeroshot
from src.open_flamingo.factory import create_model_and_transforms


class TestEvaluations(unittest.TestCase):
    def test_zeroshot_text_vqa(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")

        print(evaluate_text_vqa(model, tokenizer,
              image_processor, batch_size=1, num_samples=2))

    def test_zeroshot_imagenet(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")

        print(evaluate_imagenet_zeroshot(model, tokenizer,
              image_processor, batch_size=2, num_samples_per_class=1, num_classes=2))


if __name__ == '__main__':
    unittest.main()
