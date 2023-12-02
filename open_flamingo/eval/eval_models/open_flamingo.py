from typing import List, Dict

from PIL import Image
import torch
from einops import repeat

from open_flamingo.eval.eval_models.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.eval.utils import unwrap_model
from open_flamingo.src import VLMOutputWithPast


class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation."""

    def __init__(self, model_args, init_on_device=False):
        super().__init__(model_args, init_on_device)
        # initialize the model
        with self.init_ctx:
            (
                self.model,
                self.image_processor,
                self.tokenizer,
            ) = create_model_and_transforms(
                clip_vision_encoder_path=model_args["vision_encoder_path"],
                clip_vision_encoder_pretrained=model_args["vision_encoder_pretrained"],
                lang_model_path=model_args["lm_path"],
                tokenizer_path=model_args["tokenizer_path"],
                model_family=model_args["model_family"],
                cross_attn_every_n_layers=int(
                    model_args.get("cross_attn_every_n_layers", 1)
                ),
            )

        # load the checkpoint
        if "checkpoint_path" in model_args:
            checkpoint = torch.load(model_args["checkpoint_path"], map_location="cpu")
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint, strict=False)

        self._check_init()

    @property
    def required_args(self):
        """Return list of required arguments to initialize model."""
        return [
            "vision_encoder_path",
            "model_family",
            "lm_path",
            "tokenizer_path",
            "cross_attn_every_n_layers",
            "vision_encoder_pretrained",
        ]

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        use_cache: bool = False,
    ):
        # standard forward pass
        if past_key_values is None:
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    past_media_locations=past_media_locations,
                    past_vision_tokens=past_vision_tokens,
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
                    vision_x=vision_x,
                    lang_x=_lang_x,
                    attention_mask=_attention_mask,
                    past_key_values=past_key_values,
                    past_media_locations=past_media_locations,
                    past_vision_tokens=past_vision_tokens,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            past_media_locations = outputs.past_media_locations
            past_vision_tokens = outputs.past_vision_tokens
            logits.append(outputs.logits)

        logits = torch.cat(logits, dim=1)
        return VLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
        )

    def prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
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
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask.bool()

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        **decode_kwargs,
    ) -> List[str]:
        batch_images = self.prepare_images(batch_images)  # (B, T, 1, C, H, W)
        input_ids, attention_mask = self.prepare_text(batch_text)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    vision_x=batch_images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    **decode_kwargs,
                )

        # Extract only the new generated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool = True,
        normalize_length: bool = False,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        Note: if all classnames are one token, this code is inefficient, since we could
        get all logits after one pass. However, if there are multi-token classnames,
        we need to loop through each classname separately.
        """
        batch_images = self.prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self.prepare_text(batch_text)

        # Cache the context
        if use_cache:
            with torch.inference_mode():
                precomputed = self.__call__(
                    vision_x=batch_images,
                    lang_x=ctx_input_ids,
                    attention_mask=ctx_attention_mask,
                    use_cache=True,
                )

        # Loop through class names and get log-likelihoods
        overall_probs = []
        for class_name in all_class_names:
            # Tokenize only the class name
            classname_tokens = self.tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            num_tokens_in_classname = classname_tokens.shape[1]

            # Concatenate the class name tokens
            if not use_cache:
                _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )
                _vision_x = batch_images
            else:
                _lang_x = classname_tokens
                _attention_mask = None
                _vision_x = None

            # Call forward to get the logits
            with torch.inference_mode():
                outputs = self.__call__(
                    vision_x=_vision_x,
                    lang_x=_lang_x,
                    attention_mask=_attention_mask,
                    past_key_values=precomputed.past_key_values,
                    past_media_locations=precomputed.past_media_locations,
                    past_vision_tokens=precomputed.past_vision_tokens,
                    use_cache=False,
                )

            # Get the logits of the classname
            # logits shape is either (B, num_tokens_in_classname, vocab_len) with use_cache
            # or (B, len(_lang_x), vocab_len) without use_cache
            # remember that the logits at index t on dim 1 correspond to predictions for the t+1st token
            if use_cache:
                logits = torch.cat([precomputed.logits, outputs.logits], dim=1)

            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs[
                :, -num_tokens_in_classname - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
            gen_probs = torch.gather(
                gen_probs, 2, classname_tokens[:, :, None]
            ).squeeze(-1)

            # Aggregate over tokens in the classname
            if normalize_length:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
            overall_probs.append(class_prob)  # (B, 1)

        overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        return overall_probs

    def get_vqav2_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_ok_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_vizwiz_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_textvqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_coco_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_flickr30_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def get_hateful_memes_prompt(self, text, label=None) -> str:
        return f"<image>is an image with: '{text}' written on it. Is it hateful? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
    
    def get_waterbirds_prompt(self, label=None) -> str:
        return f"<image>Question: Is this a landbird or waterbird? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

    def get_camelyon17_prompt(self, label=None) -> str:
        return f"<image>Question: Is this a normal tissue or cancer tissue? Answer:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"