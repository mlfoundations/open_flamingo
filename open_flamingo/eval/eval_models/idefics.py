from typing import List, Dict

from PIL import Image
import torch
from einops import repeat

from models.eval_model import BaseEvalModel
from transformers import IdeficsForVisionText2Text, AutoProcessor
from transformers.models.idefics.processing_idefics import (
    image_attention_mask_for_packed_input_ids,
    incremental_to_binary_attention_mask,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import unwrap_model


class EvalModel(BaseEvalModel):
    """IDEFICS model evaluation."""

    def __init__(self, model_args, init_on_device=False):
        super().__init__(model_args, init_on_device)
        with self.init_ctx:
            self.model = IdeficsForVisionText2Text.from_pretrained(
                model_args["lm_path"]
            )
            self.processor = AutoProcessor.from_pretrained(model_args["processor_path"])
            self.tokenizer = self.processor.tokenizer
        self._check_init()

    @property
    def required_args(self):
        return ["lm_path", "processor_path"]

    def prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        batch_images = self.processor(batch)["pixel_values"]
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
        # check to see if there any <image> without <fake_token_around_image> wrapping it
        for i, text in enumerate(batch):
            if "<image>" in text and "<fake_token_around_image>" not in text:
                print(
                    "Warning: missing <fake_token_around_image> in text; inserting automatically."
                )
                batch[i] = text.replace(
                    "<image>",
                    "<fake_token_around_image><image><fake_token_around_image>",
                )

        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask

    def _compute_image_attention_mask(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """
        From: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/idefics/processing_idefics.py
        """
        max_num_images = torch.max(torch.sum(batch_tokens == 32001, dim=-1)).item()
        at_least_one_image = max_num_images > 0
        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(
                batch_tokens, self.tokenizer
            )
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, num_classes=max_num_images
            )
        else:
            # in full language mode we set the image mask to all-0s
            image_attention_mask = torch.zeros(
                batch_tokens.shape[0], batch_tokens.shape[1], 1, dtype=torch.bool
            )
        return image_attention_mask

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        batch_images = self.prepare_images(batch_images)
        input_ids, attention_mask = self.prepare_text(batch_text)
        image_attention_mask = self._compute_image_attention_mask(input_ids)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    pixel_values=batch_images,
                    image_attention_mask=image_attention_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **decode_kwargs,
                )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        use_cache: bool = False,
    ):
        image_attention_mask = self._compute_image_attention_mask(lang_x)

        # standard forward pass
        if past_key_values is None:
            with self.autocast():
                outputs = self.model(
                    pixel_values=vision_x,
                    image_attention_mask=image_attention_mask,
                    input_ids=lang_x,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            return outputs

        # loop to handle updating past_key_values
        logits = []
        for token_idx in range(lang_x.shape[1]):
            _lang_x = lang_x[:, token_idx].reshape((-1, 1))
            if attention_mask is not None:
                _attention_mask = attention_mask[:, token_idx].reshape((-1, 1))
            else:
                _attention_mask = None

            with self.autocast():
                outputs = self.model(
                    pixel_values=vision_x,
                    image_attention_mask=image_attention_mask,
                    input_ids=_lang_x,
                    attention_mask=_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def get_vqav2_prompt(self, question, answer=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Question:{question} Answer: {answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_ok_vqa_prompt(self, question, answer=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Question:{question} Answer: {answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_vizwiz_prompt(self, question, answer=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Question:{question} Answer: {answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_textvqa_prompt(self, question, answer=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Question:{question} Answer: {answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_coco_prompt(self, caption=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Caption: {caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_flickr30_prompt(self, caption=None) -> str:
        # TODO: handle prefix prompts
        return f"<image>Caption: {caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
