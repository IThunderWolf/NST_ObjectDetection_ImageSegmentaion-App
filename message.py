import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Model setup ---
@st.cache_resource
def load_vgg19_model():
    """Load the pre-trained VGG19 model"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv2'] #changed style layers

    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]

    model = tf.keras.models.Model(inputs=vgg.input, outputs=content_outputs + style_outputs)
    return model

# --- Preprocessing functions ---
def preprocess_image(image, max_dim=512):
    """Preprocess image for VGG19"""
    image = image.resize((max_dim, max_dim))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def deprocess_img(img):
    """Reverse the VGG19 preprocessing"""
    img = img[0]
    img = img + [103.939, 116.779, 123.68]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# --- Feature extraction ---
def get_features(image, model):
    """Extract content and style features"""
    outputs = model(image)
    content_features = outputs[:1]
    style_features = outputs[1:]
    return content_features, style_features

# --- Gram matrix for style loss ---
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(vector, vector, transpose_a=True)
    return gram / tf.cast(tf.shape(vector)[0], tf.float32)

# --- Loss functions ---
def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def compute_style_loss(style, generated):
    style_gram = gram_matrix(style)
    gen_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.square(style_gram - gen_gram))

# --- Style Transfer ---
def apply_nst(content_img, style_img, model, content_weight=1e4, style_weight=1e2, epochs=300):
    """Apply Neural Style Transfer"""
    generated_img = tf.Variable(content_img, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    content_features, style_features = get_features(content_img, model)
    progress_bar = st.progress(0)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            gen_content, gen_style = get_features(generated_img, model)

            # Loss calculation
            content_loss = compute_content_loss(content_features[0], gen_content[0])
            style_loss = sum([compute_style_loss(style_features[i], gen_style[i])
                                for i in range(len(style_features))])

            total_loss = content_weight * content_loss + style_weight * style_loss

        # Update image
        grad = tape.gradient(total_loss, generated_img)
        optimizer.apply_gradients([(grad, generated_img)])
        generated_img.assign(tf.clip_by_value(generated_img, 0, 255))
        progress_bar.progress((i + 1) / epochs)

    return deprocess_img(generated_img.numpy())

# --- Streamlit UI ---
st.title("Dynamic Neural Style Transfer App")
st.sidebar.header("Upload Images and Settings")

# Upload images
content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "png"])

# Settings
max_dim = st.sidebar.slider("Max Image Dimension", 256, 1024, 512)
epochs = st.sidebar.slider("Epochs", 50, 500, 300)
content_weight = st.sidebar.slider("Content Weight", 1e2, 1e5, 1e4) #changed default content weight.
style_weight = st.sidebar.slider("Style Weight", 1e1, 1e5, 1e2) #changed max style weight.

# Load VGG19 model
model = load_vgg19_model()

if content_file and style_file:
    content_img = Image.open(content_file)
    style_img = Image.open(style_file)

    st.image([content_img, style_img], caption=["Content Image", "Style Image"], width=300)

    if st.button("Apply Style Transfer"):
        # Preprocess images
        content = preprocess_image(content_img, max_dim)
        style = preprocess_image(style_img, max_dim)

        # Apply NST dynamically
        result_img = apply_nst(content, style, model, content_weight=content_weight, style_weight=style_weight, epochs=epochs)

        # Display final result
        st.image(result_img, caption="Styled Image", use_column_width=True)
        st.success("Style transfer complete!")

        # Download link
        img_pil = Image.fromarray(result_img)
        img_pil.save("styled_image.png")
        st.download_button("Download Styled Image", img_pil.tobytes(), "styled_image.png")

    print(tf.config.list_physical_devices())
    if tf.test.gpu_device_name():
        print('GPU found: {}'.format(tf.test.gpu_device_name()))
    else:
        print("No GPU found")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))