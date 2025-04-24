import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("Interactive Image Kernel Playground")

# Default kernels
default_kernels = {
    "Custom": np.zeros((3, 3)),
    "Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

input_mode = st.radio("Select Input Mode", ["Upload Image", "Webcam"])

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        img_array = np.array(image)
else:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_array = frame
        stframe.image(img_array, caption="Webcam Input", use_column_width=True)
    cap.release()

if 'img_array' in locals():
    st.image(img_array, caption="Original Image", use_column_width=True)

    # Kernel selection
    kernel_type = st.selectbox("Select a kernel", list(default_kernels.keys()))
    kernel = default_kernels[kernel_type].copy()

    if kernel_type == "Custom":
        st.subheader("Edit 3x3 Kernel")
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                kernel[i, j] = cols[j].number_input(f"({i},{j})", value=float(kernel[i, j]), key=f"{i}{j}")

    show_math = st.checkbox("Show Convolution Math (center pixel)")

    if st.button("Apply Kernel"):
        filtered = cv2.filter2D(img_array, -1, kernel.astype(np.float32))
        st.image(filtered, caption="Filtered Image", use_column_width=True)

        if show_math:
            h, w = img_array.shape
            i, j = h // 2, w // 2  # center
            region = img_array[i-1:i+2, j-1:j+2]
            if region.shape == (3, 3):
                st.subheader("Convolution Math (center pixel):")
                equation = " + ".join(
                    [f"{region[m, n]}×{round(kernel[m, n], 3)}" for m in range(3) for n in range(3)]
                )
                result = np.sum(region * kernel)
                st.markdown(f"$$\\text{{Result}} = {equation} = {round(result, 2)}$$")




