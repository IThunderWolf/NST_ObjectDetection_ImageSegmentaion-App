import streamlit as st
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import numpy as np
import segmentation_models_pytorch as smp
import cv2  # Needed for YOLO result conversion
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Vision App", page_icon="📷", layout="wide")

# ------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------
st.sidebar.title("🚀 AI Vision App")
st.sidebar.markdown("Select a task from below:")

app_mode = st.sidebar.radio(
    "Choose a task:",
    ["🏠 Home", "🔍 Object Detection", "🎨 Image Segmentation", "🖌️ Neural Style Transfer"]
)

# ------------------------------------------------------------------
# HOME PAGE (Original Styling Restored)
# ------------------------------------------------------------------
def home_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🌟 AI Vision App 🌟</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>Explore AI-powered image processing</h3>", 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Object Detection", use_container_width=True):
            st.session_state["app_mode"] = "🔍 Object Detection"
    
    with col2:
        if st.button("🎨 Image Segmentation", use_container_width=True):
            st.session_state["app_mode"] = "🎨 Image Segmentation"
    
    with col3:
        if st.button("🖌️ Neural Style Transfer", use_container_width=True):
            st.session_state["app_mode"] = "🖌️ Neural Style Transfer"

# ------------------------------------------------------------------
# OBJECT DETECTION with YOLOv8 (Fixed)
# ------------------------------------------------------------------
def object_detection():
    st.title("🔍 Object Detection - YOLOv8")
    
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        
        # ✅ Check if YOLO model exists
        model_path = "yolov8_weights.pt"
        if not os.path.exists(model_path):
            st.error(f"❌ YOLO model file '{model_path}' not found!")
            return
        
        # Load model
        model = YOLO(model_path)
        results = model(image)

        # ✅ Convert YOLO results to display correctly
        detected_img = results[0].plot()
        detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        st.image(detected_img, caption="📌 Detection Results", use_column_width=True)

# ------------------------------------------------------------------
# IMAGE SEGMENTATION (Fixed)
# ------------------------------------------------------------------
def image_segmentation():
    st.title("🎨 Image Segmentation - U-Net")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # ✅ Ensure RGB format
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        
        # Load U-Net model
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        # ✅ Check if model file exists
        state_dict_path = "best_unet_model.pth"
        if not os.path.exists(state_dict_path):
            st.error(f"❌ Model file '{state_dict_path}' not found!")
            return

        try:
            state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
            if not isinstance(state_dict, dict):
                st.error("❌ Invalid model file format!")
                return

            model.load_state_dict(state_dict)
            model.eval()

            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output = model(image_tensor)
                mask = output.squeeze().cpu().numpy()
            
            mask = (mask > 0.5).astype(np.uint8) * 255  # ✅ Convert to binary mask
            st.image(mask, caption="🖼️ Segmentation Result", use_column_width=True)

        except Exception as e:
            st.error(f"⚠️ Error during segmentation: {e}")

# ------------------------------------------------------------------
# NEURAL STYLE TRANSFER (NST) (Fixed)
# ------------------------------------------------------------------
from message import load_vgg19_model, preprocess_image, apply_nst
import tensorflow_hub as hub
import tensorflow as tf

# Load Magenta NST model from TensorFlow Hub
@st.cache_resource
def load_magenta_model():
    model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    return hub.load(model_url)

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize for consistency
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def apply_magenta_nst(content, style, model, style_weight=0.5):
    content_tensor = tf.convert_to_tensor(content, dtype=tf.float32)
    style_tensor = tf.convert_to_tensor(style, dtype=tf.float32)

    # Generate stylized image
    stylized_image = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    # Blend the original content image with the stylized result
    blended_image = (1 - style_weight) * content_tensor + style_weight * stylized_image

    return (blended_image.numpy() * 255).astype(np.uint8)


def neural_style_transfer():
    st.title("Neural Style Transfer with Magenta")
    content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png"])
    
    if content_file and style_file:
        model = load_magenta_model()
        content_img = Image.open(content_file)
        style_img = Image.open(style_file)
        st.image([content_img, style_img], caption=["Content Image", "Style Image"], width=300)

        if st.button("Apply Style Transfer"):
            content = preprocess_image(content_img)
            style = preprocess_image(style_img)
            result_img = apply_magenta_nst(content, style, model)
            st.image(result_img, caption="Styled Image", use_column_width=True)



# ------------------------------------------------------------------
# ROUTING: Select the Functionality Based on Sidebar
# ------------------------------------------------------------------
if app_mode == "🏠 Home":
    home_page()
elif app_mode == "🔍 Object Detection":
    object_detection()
elif app_mode == "🎨 Image Segmentation":
    image_segmentation()
elif app_mode == "🖌️ Neural Style Transfer":
    neural_style_transfer()



## cd C:\Users\Yatha\OneDrive\Documents\Sem VI\CVA\NST_ObjDet_ImgSeg_StreamlitApp
## streamlit run NST_StreamlitApp.py



