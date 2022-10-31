import unittest

from src.eval.evaluate import evaluate_text_vqa
from src.open_flamingo.factory import create_model_and_transforms


class TestEvaluations(unittest.TestCase):
    def test_zeroshot_text_vqa(self):
        model, image_processor, tokenizer = create_model_and_transforms(
            "openai/clip-vit-base-patch32", "facebook/opt-125m")

        print(evaluate_text_vqa(model, tokenizer,
              image_processor, batch_size=1, num_samples=10))


if __name__ == '__main__':
    unittest.main()
