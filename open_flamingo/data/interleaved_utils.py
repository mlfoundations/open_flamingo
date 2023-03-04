

import re
import torch

from .image_text_utils import preprocess_image



MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

def remove_consecutive_spaces(sentence):
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

def remove_multiple_newline_limiters(sentence):
    sentence =  re.sub(r" +", " ", sentence)
    return sentence

def preprocess_sentence(sentence):
    sentence = remove_consecutive_spaces(sentence)
    sentence = remove_multiple_newline_limiters(sentence)
    return sentence

def remove_punctuation_based_sentences(interleaved_list, is_image_list):

    """
    Sets sentences that just punctuations to empty string
    """
    for i, (interleaved_input, is_image) in enumerate(zip(interleaved_list, is_image_list)):
        if not is_image:
            is_punctuation = re.match(r"^\W+$", interleaved_input)
            if is_punctuation:
                interleaved_list[i] =  ""
    return interleaved_list, is_image_list



def get_image(input, images):
    image_name = input.split(".")[-2]+"."+input.split(".")[-1] 
    return images[image_name]

def is_tiny_image(image):
    """
    Removes images that are smaller in size for example RSS icons
    """
    return image.size[0] <= TINY_IMAGE_SIZE_THRESHOLD or image.size[1] <= TINY_IMAGE_SIZE_THRESHOLD

def remove_unwanted_images(interleaved_list, is_image_list, images):
    """
    Smaller images are usually asscoiated with icons/ advertisements and may not be relevant to the input text. 
    To tackle this issue, the method updates the interleaved data and the is_image list if the images are too small
    We also do not want to include more than five images in any interleaved sample so this method takes care of that too. 
    """
    valid_images = 0
    for i, (interleaved_input, is_image) in enumerate(zip(interleaved_list, is_image_list)):
        if is_image:
            interleaved_list[i] = get_image(interleaved_input, images)
            is_unwanted = is_tiny_image(interleaved_list[i])
            if not is_unwanted:
                valid_images +=1

            if is_unwanted or valid_images > MAX_NUM_IMAGES:
                interleaved_list[i] = ""
                is_image_list[i] = 0

    num_images = min(MAX_NUM_IMAGES, valid_images)
    return num_images, interleaved_list, is_image_list


def prepare_text_data(interleaved_list, text_tokenizer):
    """
    The method prepares text tensor
    """
    text = "".join(interleaved_list)
    text = f"{text}<|endofchunk|>{text_tokenizer.eos_token}"
    print("text...", text)
    text_tokenizer.padding_side = "right"
    text_tensor = text_tokenizer(text, max_length=MAX_NUM_TOKENS, truncation=True, padding="max_length", return_tensors="pt")
    return text_tensor


def prepare_image_data(image_list, image_processor):
    images_tensor = preprocess_image(image_list, image_processor) 
    num_images = len(image_list)
    print("num_images", num_images)
    if num_images < MAX_NUM_IMAGES:
        zero_padding = torch.zeros((MAX_NUM_IMAGES - num_images, N_CHANNELS, INTERLEAVED_IMAGE_SIZE, INTERLEAVED_IMAGE_SIZE), dtype=torch.float)
        images_tensor = torch.cat((images_tensor, zero_padding), dim=0)
    return images_tensor


def substitute_with_image_tag(interleaved_list, is_image_list, images):
    """
    The method creates a list of images (PIL) format and updates interleaved_list
    with <image> tags.
    Returns: A list of images and the updated interleaved_list list with samples

    Examples: 
    [<PIL_image>, <PIL_image>, "test sentence"]                                     ---> ["<image>", "<image>", "test sentence"]
    [<PIL_image>, <PIL_image>, "test sentence 1", "test sentence 2"]                ---> ["<image>", "<image>", "test sentence 1<|endofchunk|>", "test sentence 2"]
    [<PIL_image>, <PIL_image>, "test sentence 1", <PIL_image>, "test sentence 2"]   ---> ["<image>", "<image>", "test sentence 1<|endofchunk|>", "<image>", "test sentence 2"]
    """
    images = []
    for i, (interleaved_input, is_image) in enumerate(zip(interleaved_list, is_image_list)):
        if is_image:
            images.append(interleaved_input)
            interleaved_list[i] =  f"<image>"
        else:
            interleaved_list[i] = interleaved_list[i].strip()
            is_previous_image = interleaved_list[i-1] == f"<image>"
            is_last_sentence = i == (len(interleaved_list) - 1)
            if is_previous_image and not is_last_sentence:
                interleaved_list[i] =  f"{interleaved_list[i]}<|endofchunk|>"
            

    assert len(images) > 0, "images should be >= 1"
    return images, interleaved_list, is_image_list


def filter_out_empty_sentences(interleaved_list, is_image_list):
    filtered_interleaved_list = []
    filtered_is_image_list = []
    for i, (interleaved_input, is_image) in enumerate(zip(interleaved_list, is_image_list)):
        if not interleaved_input == "":
            filtered_interleaved_list.append(interleaved_input)
            filtered_is_image_list.append(is_image)
    return filtered_interleaved_list, filtered_is_image_list


def preprocess_sentences(interleaved_list, is_image_list):
    for i, (interleaved_input, is_image) in enumerate(zip(interleaved_list, is_image_list)):
        if not is_image:
            interleaved_list[i] = preprocess_sentence(interleaved_input)
    interleaved_list, is_image_list = remove_punctuation_based_sentences(interleaved_list, is_image_list)
    interleaved_list, is_image_list = filter_out_empty_sentences(interleaved_list, is_image_list)
    assert len(interleaved_list) == len(is_image_list) , "lengths of the interleaved and is_image list should be same"
    return len(interleaved_list), interleaved_list, is_image_list


def preprocess_interleaved_sample(interleaved_list, is_image_list, images, text_tokenizer, image_processor):
    num_images, interleaved_list, is_image_list = remove_unwanted_images(interleaved_list, is_image_list, images)
    if num_images == 0:
        raise ValueError("No images in sample")

    num_sentences, interleaved_list, is_image_list = preprocess_sentences(interleaved_list, is_image_list)
    if num_sentences == 0:
        raise ValueError("No sentences in sample")

    images, interleaved_list, is_image_list = substitute_with_image_tag(interleaved_list, is_image_list, images)

    text_tensor = prepare_text_data(interleaved_list, text_tokenizer)
    images_tensor = prepare_image_data(images, image_processor) 
    return images_tensor, (text_tensor["input_ids"], text_tensor["attention_mask"])



def preprocess_interleaved_data(data, text_tokenizer, image_processor):
    sample = data["json"]
    interleaved_list = sample["interleaved_list"]
    is_image_list = sample["is_image"]

    images = {k: v for k, v  in data.items() if k.endswith("png") or k.endswith("jpg") or k.endswith("jpeg")}
    images_tensor, (text_input_ids, text_attention_mask) = preprocess_interleaved_sample(interleaved_list, is_image_list, images, text_tokenizer, image_processor)
    return images_tensor, (text_input_ids, text_attention_mask)