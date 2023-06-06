import argparse
from typing import List

from PIL import Image
import torch

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms
from contextlib import suppress


class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, args: List[str]):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
        parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
        parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
        parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
        parser.add_argument("--checkpoint_path", type=str)
        parser.add_argument(
            "--precision",
            choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
            default="fp32",
            help="Floating point precision.",
        )        
        parser.add_argument(
            "--cross_attn_every_n_layers",
            type=int,
            default=1,
            help="how often to add a cross-attention layer after each transformer layer",
        )
        parser.add_argument("--device", type=int, default=0)
        args = parser.parse_args(args)

        # load model
        self.device = args.device if args.device >= 0 else "cpu"
        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            args.vision_encoder_path,
            args.vision_encoder_pretrained,
            args.lm_path,
            args.lm_tokenizer_path,
            cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        )
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)

        # autocast
        self.autocast =  get_autocast(args.precision)
        self.cast_dtype = get_cast_dtype(args.precision)


    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
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
        self.model.eval()

        self.tokenizer.padding_side = "left"
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            with self.autocast():
                outputs = self.model.generate(
                    self._prepare_images(batch_images).to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    attention_mask=attention_mask.to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def classification_prompt(self, class_str=None) -> str:
        return f"<image>A photo of a {class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
