import streamlit as st
from PIL import Image
from transformers import VitImageProcessor, VitForImageClassification
import torch

# Set page config
st.set_page_config (
    page_title="Image Classification App",
    page_icon="camara",
    layout="wide"
)



