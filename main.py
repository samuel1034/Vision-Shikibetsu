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
st.sidebar.write("Made with love ❤️ by Samuel Fuentes")

#File Uploader
uploaded_file = st.file_uploader(
    "Choose an Image ...",
    type=["jpg","jpeg", "png"],
    help="Upload an image file (jpeg,png)"
)

if uploaded_file is not None:
    #Display the upload image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    #Classify the image
    with st.spinner("Classifying Image"):
        try:
            #Preprocess the image
            inputs = processor(images=image, return_tensors="pt")

            #Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            #Get predicted class
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
            confidence = torch.nn.functional.softmax(logits, dim=1) [0,predicted_class_idx].item()

            #Display results
            st.success("Classification Results")
            st.write(f"**Predicted Label:** {predicted_label}")
            st.write(f"**Confidence score:** {confidence:.2%}")

            #Show top 5 predictions
            st.subheader("Top 5 predictions")
            probs = torch.nn.functional.softmax(logits, dim=1) [0]
            top5_probs, top5_indices = torch.topk(probs,5)

            for i in range(5):
                label = model.config.id2label[top5_indices[i].item()]
                prob = top5_probs[i].item()
                st.write(f"{i+1}. {label} ({prob:.2%})")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

#Add some examples
st.subheader("Examples Images to Try")
col1,col2, col3 = st.columns(3)
with col1:
    st.image("https://www.thesprucepets.com/thmb/ajN-cwmHvNjnAZW4RsIfc8yDFl8=/3600x0/filters:no_upscale():strip_icc()/dog-breeds-with-blue-eyes-5089039-01-27c7ad9283154896a8b59ecdbc911fcc.jpg",
             caption="Example Dog", width=200)
with col2:
    st.image("https://th.bing.com/th/id/OIP.i-HIGbfQ6cQIt5a7RipD4wHaHa?cb=iwp1&rs=1&pid=ImgDetMain",
             caption="Example: Cat", width=200)

with col3:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/90/Labrador_Retriever_portrait.jpg",
             caption="Example: Labrador Retriever", width=200)

















