from time import sleep

import requests
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from open_flamingo import create_model_and_transforms


@st.experimental_singleton
def get_model():
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo3B-v0", "checkpoint_v2.pt")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="openai/clip-vit-large-patch14",
        clip_processor_path="openai/clip-vit-large-patch14",
        lang_encoder_path="facebook/opt-1.3b",
        tokenizer_path="facebook/opt-1.3b",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state_dict"] = {
        k.replace("gated_cross_attn_layers.", "lang_encoder.gated_cross_attn_layers."): v
        for k, v in checkpoint["model_state_dict"].items()
    }

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, image_processor, tokenizer


# only load model once
model, image_processor, tokenizer = get_model()

st.title("OpenFlamingo Demo")
st.write(
    "This is a demo of the OpenFlamingo model. It is a model that can generate text from images and text. The current model is based on a CLIP and OPT backbone. As with any language model, beware of occasional offensive text."
)

task = st.selectbox("Select a task", ["Captioning 🗯️", "Visual Question Answering 🙋"])

if task == "Captioning 🗯️":
    st.write("This is a captioning task. You can upload an image and the model will generate a caption for it.")
    if st.button("Try a random image", type="primary"):
        st.session_state["image"] = Image.open(requests.get("https://source.unsplash.com/random", stream=True).raw)
    elif st.session_state.get("image") is None:
        st.session_state["image"] = st.file_uploader("Or upload an image", type=["jpg", "png"])

    if st.session_state["image"]:
        st.image(st.session_state["image"], width=300)

    if st.button("Generate caption", type="primary"):
        with st.spinner("Generating caption..."):
            input_ids = tokenizer("<image>", return_tensors="pt")["input_ids"]
            output = model.generate(
                image_processor(images=[st.session_state["image"]], return_tensors="pt")["pixel_values"]
                .unsqueeze(1)
                .unsqueeze(1),
                input_ids,
                max_length=25,
            )
            st.success(tokenizer.decode(output[0][len(input_ids[0]) :], skip_special_tokens=True))

if task == "Visual Question Answering 🙋":
    st.write(
        "This is a visual question answering task. You can upload an image and the model will generate an answer for it."
    )
    if st.button("Try a random image", type="primary"):
        st.session_state["image"] = Image.open(requests.get("https://source.unsplash.com/random", stream=True).raw)
    elif st.session_state.get("image") is None:
        st.session_state["image"] = st.file_uploader("Or upload an image", type=["jpg", "png"])

    if st.session_state["image"]:
        st.image(st.session_state["image"], width=300)

    question = st.text_input("Question", value="What is this?")

    if st.session_state["image"] and st.button("Generate answer", type="primary"):
        with st.spinner("Generating answer..."):
            input_ids = tokenizer("<image>" + "question:" + question + " answer:", return_tensors="pt")["input_ids"]
            output = model.generate(
                image_processor(images=[st.session_state["image"]], return_tensors="pt")["pixel_values"]
                .unsqueeze(1)
                .unsqueeze(1),
                input_ids,
                max_length=100,
            )
            st.success(tokenizer.decode(output[0][len(input_ids[0]) :], skip_special_tokens=True))
if st.button("Reset"):
    for key in st.session_state:
        del st.session_state[key]
    st.experimental_rerun()
