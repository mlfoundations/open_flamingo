from typing import List

from PIL import Image
import torch

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from open_flamingo.eval.eval_models import BaseEvalModel
from open_flamingo.eval.utils import unwrap_model


class EvalModel(BaseEvalModel):
    """BLIP-2 model evaluation."""

    def __init__(self, model_args, init_on_device=False):
        super().__init__(model_args, init_on_device)
        with self.init_ctx:
            self.processor = Blip2Processor.from_pretrained(model_args["processor_path"])
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_args["lm_path"]
            )
            self.tokenizer = self.processor.tokenizer
        self._check_init()

    @property
    def required_args(self):
        return ["processor_path", "lm_path"]

    def prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        batch_images = None
        assert all(
            len(example) == 1 for example in batch
        ), "BLIP-2 only supports one image per example"
        for example in batch:
            if batch_images is None:
                batch_images = self.processor.image_processor(
                    example, return_tensors="pt"
                )["pixel_values"]
            else:
                batch_images = torch.cat(
                    [
                        batch_images,
                        self.processor.image_processor(example, return_tensors="pt")[
                            "pixel_values"
                        ],
                    ]
                )
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images

    def prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
        add_special_tokens=True,
    ):
        self._validate_text(batch)
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        batch_images = self.prepare_images(batch_images)  # (B, C, H, W)
        input_ids, attention_mask = self.prepare_text(batch_text)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    batch_images,
                    input_ids,
                    attention_mask=attention_mask,
                    **decode_kwargs,
                )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_vqav2_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"

    def get_ok_vqa_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"

    def get_vizwiz_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"

    def get_textvqa_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"

    def get_coco_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"
    
    def get_flickr30_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"
