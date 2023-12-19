#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from PIL import Image
import torch
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor, AutoProcessor
from transformers import BlipConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(r"D:\Dropbox\Yueh\YU\Artificial_Intelligence\Final_Project\AI_Final_Project_Code\Code\BLIP\blip_model_v2_epo89" )


def preprocess_image(image):
    # Your image preprocessing logic here...
    # Example: Resize image to 128x128 pixels
    image = image.resize((128, 128))
    image_encoding = image_processor(image,
                                     do_resize=True,
                                     size=(128, 128),
                                     return_tensors="pt")
    return image_encoding["pixel_values"][0]

def preprocess_text(text, max_length=32):
    # Your text preprocessing logic here...
    encoding = text_processor(
        None,
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    for k, v in encoding.items():
        encoding[k] = v.squeeze()
    return encoding

def predict(image, question):
    # Preprocess image
    pixel_values = preprocess_image(image).unsqueeze(0)

    # Preprocess text
    encoding = preprocess_text(question)

    # Print shapes for debugging
    #print("Pixel Values Shape:", pixel_values.shape)
    #print("Input IDs Shape:", encoding['input_ids'].unsqueeze(0).shape)

    # Perform prediction using your model
    # Example: Replace this with your actual prediction logic
    model.eval()
    outputs = model.generate(pixel_values=pixel_values, input_ids=encoding['input_ids'].unsqueeze(0))

    prediction_result = text_processor.decode(outputs[0], skip_special_tokens=True)

    return prediction_result

def main():
    st.title("PathoAgent")

    # Image upload
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

    # Text input
    st.subheader("Input Question")
    text_input = st.text_area("Enter text here:")

    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        #resized_img = image.resize((10,10))
        st.image(image, caption="Uploaded Image.", use_column_width=True)



    # Predict button
    if st.button("Predict"):
        if uploaded_file is not None and text_input:
            # Perform prediction
            prediction_result = predict(image, text_input)

            # Display input text
            st.subheader("Input Question:")
            st.write(text_input)
            # Display prediction result
            st.subheader("Prediction Result:")
            st.write(prediction_result)

if __name__ == "__main__":
    main()


# streamlit run Streamlit.py