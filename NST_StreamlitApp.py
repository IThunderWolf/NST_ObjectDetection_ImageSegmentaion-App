import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
from ultralytics import YOLO
import torchvision.transforms as transforms

# Set page title
st.set_page_config(page_title="AI Vision App", page_icon="📷", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a task:", ["Home", "Object Detection", "Image Segmentation", "Neural Style Transfer"])

# Home Page
def home_page():
    st.title("AI Vision App")
    st.markdown("### Select a task to proceed")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Object Detection"):
            st.session_state["app_mode"] = "Object Detection"
    
    with col2:
        if st.button("🎨 Image Segmentation"):
            st.session_state["app_mode"] = "Image Segmentation"
    
    with col3:
        if st.button("🖌️ Neural Style Transfer"):
            st.session_state["app_mode"] = "Neural Style Transfer"

# Object Detection with YOLOv8
def object_detection():
    st.title("Object Detection with YOLOv8")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Load YOLOv8 model (Modify the path accordingly)
        model = YOLO("yolov8_weights.pt")  # Update with your actual weights file path
        results = model(image)
        
        st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

# Image Segmentation (CUB-200-2011 Dataset)
def image_segmentation():
    st.title("Image Segmentation - CUB-200-2011")
    uploaded_file = st.file_uploader("Upload an image from CUB-200-2011", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Load segmentation model
        model_path = "best_unet_model.pth"
        if not os.path.exists(model_path):
            st.error(f"Model file {model_path} not found. Please check the file path.")
            return
        
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            mask = output.squeeze().numpy()
        
        st.image(mask, caption="Segmentation Result", use_column_width=True)

# Neural Style Transfer (NST)
@st.cache_resource
def load_vgg19_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    return tf.keras.models.Model(inputs=vgg.input, outputs=content_outputs + style_outputs)

def preprocess_image(image, max_dim=512):
    image = image.resize((max_dim, max_dim))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def deprocess_img(img):
    img = img[0]
    img = img + [103.939, 116.779, 123.68]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def get_features(image, model):
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    outputs = model(image)
    content_features = outputs[:1]
    style_features = outputs[1:]
    return content_features, style_features

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(vector, vector, transpose_a=True)
    return gram / tf.cast(tf.shape(vector)[0], tf.float32)

def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def compute_style_loss(style, generated):
    style_gram = gram_matrix(style)
    gen_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.square(style_gram - gen_gram))

def apply_nst(content_img, style_img, model, content_weight=1e4, style_weight=1e2, epochs=300):
    generated_img = tf.Variable(content_img, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    content_features, style_features = get_features(content_img, model)
    
    for i in range(epochs):
        with tf.GradientTape() as tape:
            gen_content, gen_style = get_features(generated_img, model)
            content_loss = compute_content_loss(content_features[0], gen_content[0])
            style_loss = sum([compute_style_loss(style_features[i], gen_style[i]) for i in range(len(style_features))])
            total_loss = content_weight * content_loss + style_weight * style_loss
        grad = tape.gradient(total_loss, generated_img)
        optimizer.apply_gradients([(grad, generated_img)])
        generated_img.assign(tf.clip_by_value(generated_img, 0, 255))
    return deprocess_img(generated_img.numpy())

def neural_style_transfer():
    st.title("Neural Style Transfer")
    content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png"])
    model = load_vgg19_model()
    if content_file and style_file:
        content_img = Image.open(content_file)
        style_img = Image.open(style_file)
        st.image([content_img, style_img], caption=["Content Image", "Style Image"], width=300)
        if st.button("Apply Style Transfer"):
            content = preprocess_image(content_img)
            style = preprocess_image(style_img)
            result_img = apply_nst(content, style, model)
            st.image(result_img, caption="Styled Image", use_column_width=True)
            st.download_button("Download Styled Image", Image.fromarray(result_img).tobytes(), "styled_image.png")

# Route to appropriate function
if app_mode == "Home":
    home_page()
elif app_mode == "Object Detection":
    object_detection()
elif app_mode == "Image Segmentation":
    image_segmentation()
elif app_mode == "Neural Style Transfer":
    neural_style_transfer()


## cd C:\Users\Yatha\OneDrive\Documents\Sem VI\CVA\NST_ObjDet_ImgSeg_StreamlitApp
## streamlit run NST_StreamlitApp.py