import streamlit as st
from streamlit_chat import message as st_message
import numpy as np
import cv2
from  PIL import Image, ImageEnhance
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration

@st.experimental_singleton
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the

    #model_name = "sshleifer/tiny-gpt2"
    model_name = "facebook/blenderbot-400M-distill"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    #model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer

if "history" not in st.session_state:
    st.session_state.history = []




def generate_answer():
    tokenizer = get_models()
    user_message = st.session_state.user_text
    user_image = st.session_state.user_image
    inputs = tokenizer(user_message,truncation = True,padding = False)['input_ids']
    #result = model.generate(inputs,
    #do_sample=True,
    #max_new_tokens=10,
    #top_k=50)
    message_bot = tokenizer.decode(
        inputs, skip_special_tokens=True
    )

    st.session_state.history.append({"message" : user_message, "is_user":True,"image" : user_image})
    st.session_state.history.append({"message" : message_bot, "is_user":False})

# Webpage title
st.title(":flamingo: :blue[Open Flamingo Demo]")

# sentance 1 \n

#Add a text input for the user
#st.text_input("Ask :flamingo: anything", key = "input_text", on_change = generate_answer)


#col1, col2, col3 = st.columns(3)
#with col2:

# FOR 3 IMAGE SUPPORT
#with col2:
#    uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'],key = "img2")
#    st.text_input("Ask :flamingo:",key = "text2")
#with col3:
#    uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'],key = "img3")
 #   st.text_input("Ask :flamingo:",key = "text3"

for i,chat in enumerate(st.session_state.history):
    _, _, _, col2,_, _, _ = st.columns([2]*6+[1.20])
    if chat["is_user"] and chat["image"] is not None:
        with col2:
            st.image(chat["image"],width = 300)
    st_message(message = chat["message"],is_user = chat["is_user"],key = i)  # unpacking



uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'],key = "user_image")
if uploaded_file_1 is not None:
    user_image = Image.open(uploaded_file_1)
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = user_image.size
    # Setting the points for cropped image
    left = 4
    top = height / 5
    right = 154
    bottom = 3 * height / 5
    # Cropped image of above dimension
    # (It will not change original image)
    user_image = user_image.crop((left, top, right, bottom))
    newsize = (100, 100)
    user_image = user_image.resize(newsize)
col1, col2 = st.columns([9, 1])
with col1:
    st.text_input("Ask :flamingo:",key = "user_text")
with col2:
    st.text(" ")
    st.text(" ")

    st.button('Send',on_click=generate_answer,type = "primary")
