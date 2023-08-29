import abc
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from utils import get_autocast, get_cast_dtype
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


class BaseEvalModel(abc.ABC):
    """Base class encapsulating functionality needed to evaluate a model."""

    def __init__(self, model_args: List[str]):
        """Initialize model.

        Args:
            args: arguments to model. These should be parsed, or if the model
                has no applicable arguments, an error should be thrown if `args`
                is non-empty.
        """

    def __init__(self, model_args):
        assert "lm_path" in model_args, "All models require the lm_path argument"
        self.device = (
            model_args["device"]
            if ("device" in model_args and model_args["device"] >= 0)
            else "cpu"
        )
        precision = model_args.get("precision", "fp32")
        self.lm_name = model_args["lm_path"].split("/")[-1]
        self.autocast = get_autocast(precision)
        self.cast_dtype = get_cast_dtype(precision)

    def _check_init(self):
        """Finish model initialization."""
        assert hasattr(self, "model"), "Model has not been initialized"
        self.model.eval()
        self.model.to(self.device, dtype=self.cast_dtype)
        assert hasattr(self, "tokenizer"), "Tokenizer has not been initialized"
        self.tokenizer.padding_side = "left"

    def init_distributed(self, world_size=None, use_deepspeed=False):
        """Wrap model as DDP or deepspeed."""
        if use_deepspeed:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                self.model,
                mp_size=world_size,
                dtype=self.cast_dtype,
                checkpoint=None,
                replace_with_kernel_inject=True,
            )
            self.model = self.ds_engine.module
        else:
            self.model = DDP(self.model, device_ids=[self.device])

    def set_device(self, device):
        """Set device for model."""
        self.device = device
        self.model = self.model.to(device)

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        use_cache: bool = False,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """

    def prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
        add_special_tokens=True,
    ):
        """
        Prepare text for model.

        Args:
            batch: list of text strings
            padding: whether to pad the text
            truncation: whether to truncate the text
            max_length: maximum length of the text

        Returns:
            input_ids: tensor of shape (B, T)
            attention_mask: tensor of shape (B, T)
        """

    def prepare_images(self, batch: List[List[Image.Image]]):
        """
        Prepare images for model.
        Args:
            batch: list of lists of PIL images
        Returns:
            tensor of shape (B, T, *, C, H, W)
        """

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        """Get outputs for a batch of images and text.

        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
              where each list contains the images for a single example.
            max_generation_length: maximum length of the generated caption.
                Defaults to 10.
            num_beams: number of beams to use for beam search. Defaults to 3.
            length_penalty: length penalty for beam search. Defaults to -2.0.

        Returns:
            List of decoded output strings.
        """

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
                where each list contains the images for a single example.
            all_class_names: list of all class names.
            use_cache: whether to cache the context to speed up evaluations.
            normalize_length: whether to normalize logprobs by the length of the
                class name
        Returns:
            (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
