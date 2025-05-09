import streamlit as st
from PIL import Image
from transformers import VitImageProcessor, VitForImageClassification
import torch

# Set page config
st.set_page_config (
    page_title="Image Classification App",
    page_icon=":camara:",
    layout="wide"
)

#Load the pre-trained model and processor
@st.cache_resource
def load_model():
    processor = VitImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = VitForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return processor, model

processor, model = load_model()

#Define class labels  (Using ImageNet Classes)
with open ('iamagenet_classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# App title and description





