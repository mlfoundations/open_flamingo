
import re
import torch
from nltk import sent_tokenize

def preprocess_pile(sample, tokenizer, clip_processor):
    sample = sample[0].decode("utf-8")

    # remove multiple consecutive spaces
    sample = re.sub(r"\s+", " ", sample)
    # remove multiple newlines delimiters
    sample = re.sub(r" +", " ", sample)


    sentences = sent_tokenize(sample)
    # remove sentences that are just punctuation
    sentences = [s for s in sentences if not re.match(r"^\W+$", s)]

    if len(sentences) == 0:
        raise ValueError("No sentences in sample")

    # replace sentences 70% of the time
    indices_replaced = torch.zeros(len(sentences), dtype=torch.bool)
    indices_replaced[torch.rand(len(sentences)) <= 0.7] = True

    if indices_replaced.sum() == 0:
        raise ValueError("No sentences to mask")

    # cap the number of sentences to replace to 10
    if indices_replaced.sum() > 10:
        true_indices = torch.nonzero(indices_replaced).squeeze()
        overflowing = indices_replaced.sum() - 10
        indices_replaced[
            true_indices[torch.randperm(len(true_indices))[:overflowing]]
        ] = False

    chosen_sentences = [
        sentences[i].strip()
        for i in range(len(indices_replaced))
        if indices_replaced[i]
    ]

    for i in range(len(sentences)):
        if indices_replaced[i]:
            sentences[i] = f"<|endofchunk|><image>{sentences[i]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)
    text = text.replace(" <|endofchunk|>", "<|endofchunk|>")
    text = text.replace("<image> ", "<image>")
    text = text.replace(" <image>", "<image>")


    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    clip_text_tensor = clip_processor.tokenizer(
        chosen_sentences,
        max_length=24,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # pad to 10 sentences
    if len(chosen_sentences) < 10:
        zero_padding = torch.zeros((10 - len(chosen_sentences), 24), dtype=torch.long)
        clip_text_tensor["input_ids"] = torch.cat(
            (clip_text_tensor["input_ids"], zero_padding), dim=0
        )
        clip_text_tensor["attention_mask"] = torch.cat(
            (clip_text_tensor["attention_mask"], zero_padding), dim=0
        )

    return (clip_text_tensor["input_ids"], clip_text_tensor["attention_mask"]), (
        text_tensor["input_ids"],
        text_tensor["attention_mask"],
    )
