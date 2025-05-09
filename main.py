import streamlit as st
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
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
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return processor, model

processor, model = load_model()

#Define class labels  (Using ImageNet Classes)
with open ('imagenet_classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# App title and description
st.title("🖼️ Image Classification App")
st.write ("""
Upload an image, and this app will classify it into one of 1000 ImageNet categories.
The app uses a Vision Transformer (ViT) model from Hugging Face Transformers.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
This demo shows how to:
- Use pre-trained vision models from Hugging Face
- Create an interactive web app with Streamlit
- Deploy an AI application
""")
st.sidebar.write("[Github Repository] (https://github.com/samuel1034/Vision-Shikibetsu)")

#File Uploader
uploaded_file = st.file_uploader(
    "Choose an Image ...",
    type=["jpg","jpeg", "png"],
    help="Upload an image file (jpeg,png)"
)

if uploaded_file is not None:
    #Display the upload image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    #Classify the image
    with st.spinner("Classifying Image"):
        try:
            #Preprocess the image
            inputs = processor(image=image, return_tensors="pt")

            #Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            #Get predicted class
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
            confidence = torch.nn.functional.softmax(logits, dim=1) [0,predicted_class_idx].item()

            #













