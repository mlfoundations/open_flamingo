from typing import List

from PIL import Image
import torch

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.utils import unwrap_model


class EvalModel(BaseEvalModel):
    """BLIP-2 model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "cpu"
    """

    def __init__(self, model_args):
        assert (
            "processor_path" in model_args and "lm_path" in model_args
        ), "BLIP-2 requires processor_path, lm_path, and device arguments to be specified"

        self.processor = Blip2Processor.from_pretrained(model_args["processor_path"])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_args["lm_path"]
        )
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"
        self.lm_name = model_args["lm_path"].split("/")[-1]

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, channels, height, width).
        """
        batch_images = None
        assert all(
            len(example) == 1 for example in batch
        ), "BLIP-2 only supports one image per example"

        for example in batch:
            assert len(example) == 1, "BLIP-2 only supports one image per example"
            batch_images = torch.cat(
                [
                    batch_images,
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ],
                ]
                if batch_images is not None
                else [
                    self.processor.image_processor(example, return_tensors="pt")[
                        "pixel_values"
                    ]
                ],
                dim=0,
            )
        return batch_images

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        encodings = self.processor.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = unwrap_model(self.model).generate(
                self._prepare_images(batch_images).to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                min_new_tokens=min_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_vqa_prompt(self, question, answer=None) -> str:
        return (
            f"Question:{question} Short answer:{answer if answer is not None else ''}"
        )

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        raise NotImplementedError(
            "BLIP-2 classification-based evaluation not implemented"
        )
