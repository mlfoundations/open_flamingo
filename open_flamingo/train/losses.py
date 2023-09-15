from open_flamingo.src.vlm import VLM
import torch


class Loss:
    @property
    def name(self):
        raise NotImplementedError

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        """
        Args:
            model: VLM model
            images: images tensor, already moved to device and cast to appropriate dtype
                shape (B, T_img, F, C, H, W)
            input_ids: input ids tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            attention_mask: attention mask tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            autocast: autocast context manager
        """
        raise NotImplementedError


class NextTokenPrediction(Loss):
    @property
    def name(self):
        return "next_token_prediction"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.eos_token] = -100
        labels[
            torch.isin(labels, torch.Tensor(unwrap_model(model).special_token_ids))
        ] = -100
        labels = labels.to(input_ids.device)

        # call forward
        with autocast():
            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        return loss


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        return model.module
    else:
        return model
