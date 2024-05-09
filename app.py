import cv2
import numpy as np
import streamlit as st
import base64
import os

# Function for color quantization
def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Function for cartoonifying an image
def cartoonify(image):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Color Quantization
    img_quantized = color_quantization(imgRGB, 7)

    # Convert to grayscale
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

    # Identifying Edges
    edges = cv2.Canny(gray, 100, 200)

    # Blurring the edges
    blurred = cv2.medianBlur(img_quantized, 5)

    # Adding edges to complete the cartoonification
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon

# Function to generate a download link for the cartoonified image
def get_image_download_link(img, filename, text):
    buffered = cv2.imencode('.jpg', img)[1]
    img_str = base64.b64encode(buffered).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Streamlit UI
st.title("Cartoonify Your Image!")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Original Image', use_column_width=True)

    if st.button('Cartoonify'):
        cartoon_image = cartoonify(image)
        st.image(cartoon_image, caption='Cartoonified Image', use_column_width=True)

        # Allowing download of the cartoonified image
        st.markdown(get_image_download_link(cartoon_image, "cartoonified_image.jpg", "Download Cartoonified Image"), unsafe_allow_html=True)
