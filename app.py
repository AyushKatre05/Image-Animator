import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# Function for color quantization
def color_quantization(img, k):
    # Convert the image to RGB mode
    img_rgb = img.convert('RGB')

    # Convert the image to a NumPy array
    img_array = np.array(img_rgb)

    # Perform color quantization using k-means clustering
    pixels = img_array.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Select k random points as initial cluster centers
    np.random.seed(42)
    idx = np.random.randint(pixels.shape[0], size=k)
    centers = pixels[idx]

    # Iteratively update cluster centers
    for _ in range(10):
        # Assign each pixel to the nearest cluster center
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update cluster centers
        new_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    # Assign each pixel to the nearest cluster center
    distances = np.linalg.norm(pixels[:, np.newaxis, :] - centers, axis=2)
    labels = np.argmin(distances, axis=1)

    # Replace each pixel with its corresponding cluster center
    quantized_pixels = centers[labels]
    quantized_image = quantized_pixels.reshape(img_array.shape)

    # Convert the quantized image array back to PIL image
    quantized_image_pil = Image.fromarray(np.uint8(quantized_image))

    return quantized_image_pil

# Function for cartoonifying an image
def cartoonify(image):
    # Convert the image to grayscale
    gray = image.convert('L')

    # Apply edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # Apply color quantization
    cartoon_image = color_quantization(image, 7)

    # Resize edges image to match cartoon image dimensions
    edges = edges.resize(cartoon_image.size)

    # Convert both images to RGBA mode to allow blending with transparency
    cartoon_image = cartoon_image.convert("RGBA")
    edges = edges.convert("RGBA")

    # Create a composite image with edges and cartoon image
    composite_image = Image.alpha_composite(cartoon_image, edges)

    return composite_image

# Streamlit UI
st.title("Cartoonify Your Image!")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    if st.button('Cartoonify'):
        cartoon_image = cartoonify(image)
        st.image(cartoon_image, caption='Cartoonified Image', use_column_width=True)
