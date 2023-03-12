
import torchvision



def preprocess_image(sample, image_processor):
    image = image_processor(images=sample, return_tensors="pt")["pixel_values"]
    # apply random horizontal flip and color jitter
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    image = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(image)
    return image

def preprocess_text(sample, tokenizer):
    tokenizer.padding_side = "right"
    sample = [
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]

    text = tokenizer(
        sample,
        max_length=32,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]