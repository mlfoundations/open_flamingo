import argparse
from typing import List

from PIL import Image
import torch

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms


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
        parser.add_argument("--checkpoint_path", type=str, required=True)
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
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)

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
        return preprocessed

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
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
            outputs = self.model.generate(
                self._prepare_images(batch_images).to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
