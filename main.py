import streamlit as st
from PIL import Image
import torch
import time
import numpy as np
from transformers import ViTImageProcessor
import onnxruntime
from optimum.onnxruntime import ORTModelForImageClassification
import traceback # Import traceback

# Set page configuration
st.set_page_config(
    page_title="Vision識別 - Optimized Image Classifier",
    page_icon="🖼️",
    layout="wide"
)


# --- Cache Only the Processor ---
@st.cache_resource
def load_processor():
    """Load and cache only the image processor (pickle-safe)"""
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


# --- Initialize ONNX Model ---
def init_onnx_model():
    """Initialize ONNX model with performance optimizations"""
    # Configure session options for better performance
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    # Determine execution provider (GPU if available)
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

    # Ensure the model is loaded from the Hugging Face hub directly if not cached
    # ORTModelForImageClassification.from_pretrained handles caching internally
    try:
        # Use export=True with provider specified
        return ORTModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            export=True,
            provider=providers[0],
            session_options=sess_options
        )
    except Exception as e:
        st.error(f"Failed to load or export ONNX model: {e}")
        st.stop() # Stop execution if model loading fails


# Initialize components
# Use st.cache_resource for the model as well, as it's a heavy object
# This avoids reloading the model every time the script reruns due to interaction
@st.cache_resource
def load_onnx_model():
    return init_onnx_model()

processor = load_processor()
onnx_model = load_onnx_model()


# --- Performance Tracking ---
def track_performance(func):
    """Decorator to measure inference time and memory usage"""

    def wrapper(*args, **kwargs):
        # Start timing and memory tracking
        start_time = time.perf_counter()
        start_mem = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Memory tracking can be complex with shared resources; this is a basic attempt
            # For more accurate tracking, consider nvidia-smi or more advanced tools
            try:
                 start_mem = torch.cuda.memory_allocated()
            except Exception:
                 start_mem = 0 # Handle potential errors if CUDA is available but context is tricky

        # Run the function
        result = func(*args, **kwargs)

        # Calculate performance metrics
        st.session_state.inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        if torch.cuda.is_available() and start_mem > 0:
             try:
                 st.session_state.vram_usage = (torch.cuda.memory_allocated() - start_mem) / 1e6  # Convert to MB
             except Exception:
                  st.session_state.vram_usage = None # Indicate failure if tracking fails

        return result

    return wrapper


# --- Image Classification Function ---
@track_performance
def classify_image(image):
    """Classify an image using the ONNX model"""
    # Ensure image is in RGB format if it's grayscale or RGBA
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Process the image using the ViTImageProcessor
    inputs = processor(images=image, return_tensors="pt")

    # ONNX Runtime inference
    # ORTModelForImageClassification takes the processed inputs (PyTorch tensors)
    outputs = onnx_model(**inputs)

    # The output is typically an ORTModelOutput object, access logits
    logits = outputs.logits

    # Ensure logits is a torch tensor (it should be if using return_tensors="pt")
    if not isinstance(logits, torch.Tensor):
         # Convert to torch tensor if necessary (e.g., if output was numpy)
         logits = torch.from_numpy(logits)

    return logits


# --- App UI ---
st.title("🖼️ Vision識別 / Image Classifier")
st.write("""
Upload an image to classify it across 1,000+ ImageNet categories using an optimized
Vision Transformer with ONNX runtime acceleration.
""")

# Sidebar with system information
st.sidebar.header("System Information")
device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🐢"
st.sidebar.write(f"Device: {device}")
if torch.cuda.is_available():
    try:
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.sidebar.write(f"CUDA: {torch.version.cuda}")
    except Exception:
         st.sidebar.write("GPU details unavailable.")


st.sidebar.header("Optimizations")
st.sidebar.write("✓ ONNX Runtime")
st.sidebar.write("✓ Memory Efficient (via ONNX)")
st.sidebar.write("✓ Fast Inference")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Perform classification
        with st.spinner("Analyzing image..."):
            logits = classify_image(image) # classify_image now returns logits directly

            # Process results
            # Ensure logits is a torch tensor
            if not isinstance(logits, torch.Tensor):
                 logits = torch.from_numpy(logits) # Convert if it came as numpy

            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

            # Display results
            st.success("Classification Results")

            # Results columns
            col1, col2 = st.columns(2)
            with col1:
                # Corrected syntax for f-string with [] and .item()
                st.write(f"**Top Prediction:** {(onnx_model.config.id2label[top5_indices[0].item()])}")
                st.write(f"**Confidence:** {top5_probs[0].item():.2%}")

            with col2:
                st.metric("Inference Time", f"{st.session_state.inference_time:.1f} ms")
                if 'vram_usage' in st.session_state and st.session_state.vram_usage is not None:
                    st.metric("VRAM Usage", f"{st.session_state.vram_usage:.1f} MB")
                elif torch.cuda.is_available():
                     st.metric("VRAM Usage", "Tracking N/A")


            # Top 5 predictions with progress bars
            st.subheader("Top 5 Predictions")
            for i in range(5):
                # Access label using the corrected syntax pattern
                label = onnx_model.config.id2label[top5_indices[i].item()]
                prob = top5_probs[i].item()
                st.progress(float(prob), text=f"{i + 1}. {label} ({prob:.2%})")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        # Display full traceback for debugging
        st.error(traceback.format_exc())


# Example images section
st.subheader("Try These Examples")
st.write("*(Note: Example images below are for display only and won't trigger classification.)*")
example_cols = st.columns(3)
# --- UPDATED IMAGE URLS (linking to original files) ---
example_images = [
    ("https://www.petpaw.com.au/wp-content/uploads/2014/06/Pembroke-Welsh-Corgi-4.jpg", "Corgi"), # Direct link to a common Corgi image
    ("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg", "Cat"), # Direct link to the cat image
    ("https://media.istockphoto.com/id/122286715/photo/white-labrador-retriever-on-a-white-background.jpg?s=612x612&w=0&k=20&c=I_p1VTPM9ZUIOfDwTlbrsADjdVZ5IJrHRlob8LToCdE=", "Labrador") # Direct link to the labrador image
]

for col, (img_url, caption) in zip(example_cols, example_images):
    with col:
        st.markdown(f'<p style="text-align:center; font-size:smaller;">{caption}</p>', unsafe_allow_html=True)
        # Using use_container_width as recommended
        st.image(img_url, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: small;
    color: gray;
    text-align: center;
}
</style>
<div class="footer">
    Powered by Hugging Face Transformers and ONNX Runtime • Made with ❤️ by Samuel Fuentes
</div>
""", unsafe_allow_html=True)