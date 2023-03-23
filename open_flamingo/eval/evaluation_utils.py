from dataclasses import dataclass
import torch
from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.src.flamingo import Flamingo
from typing import Any, Tuple, Callable


@dataclass
class FlamingoModelLoader:
    """A lightweight container to load a fully-initialized Flamingo model."""
    clip_path: str
    lm_path: str
    lm_tokenizer_path: str
    checkpoint_path: str

    def load(self, device: int) -> Tuple[Flamingo, Any, Any]:
        """Instantiate the model, load from the checkpoint, and return it."""
        flamingo, image_processor, tokenizer = create_model_and_transforms(
            self.clip_path,
            self.clip_path,
            self.lm_path,
            self.lm_tokenizer_path)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")[
            "model_state_dict"]
        # remove the "module." prefix from the keys
        checkpoint = {k.replace("module.", ""): v for k, v in
                      checkpoint.items()}

        flamingo.load_state_dict(checkpoint, strict=False)
        flamingo.to(device if device >= 0 else "cpu")

        return flamingo, image_processor, tokenizer


def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = image_processor(
            images=[s["image"] for s in in_context_samples],
            return_tensors="pt",
        )["pixel_values"]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_context_text(get_prompt: Callable[[dict], str], in_context_samples,
                     effective_num_shots, num_shots) -> str:
    context_text = (
        "".join([get_prompt(s) for s in in_context_samples]
                ) if effective_num_shots > 0 else ""
    )

    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def prepare_batch_images(batch, image_processor, context_images,
                         num_shots) -> torch.Tensor:
    """Helper function to prepare images from a batch.

    Args:
        batch: the batch of inputs.
        image_processor: the image processor.
        context_images: context images to prepend to the batch image.
        num_shots: number of shots; should be identical to number of context
            images.
    Returns:
        Tensor of shape [batch_size, num_shots + 1, 1, num_channels, h, w].
    """
    batch_images = None
    for b in batch:
        b_image = image_processor(images=[b["image"]], return_tensors="pt")[
            "pixel_values"
        ]
        b_image = b_image.unsqueeze(1).unsqueeze(0)
        b_image = (
            torch.cat([context_images, b_image], dim=1)
            if num_shots > 0
            else b_image
        )

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images
