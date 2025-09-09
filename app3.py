import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from rembg import remove
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from typing import Union
import threading

# --- Image Effect Functions ---

def apply_effect(frame: np.ndarray, effect_type: str) -> np.ndarray:
    """Applies a selected visual effect to an image frame."""
    if effect_type == "Grayscale":
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2BGR)
    
    elif effect_type == "Warm":
        return apply_warm_effect(frame)
    
    elif effect_type == "Cold":
        return apply_cold_effect(frame)
    
    elif effect_type == "Sepia":
        return apply_sepia_effect(frame)
    
    elif effect_type == "Sketch":
        return apply_sketch_effect(frame)
    
    # Return the original frame if no effect is selected
    return frame

def apply_sepia_effect(bgr_image: np.ndarray) -> np.ndarray:
    """Applies a sepia tone effect to a BGR image."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32) / 255
    
    # Create a sepia color layer
    sepia = np.ones(bgr_image.shape)
    sepia[:, :, 0] *= 153  # Blue channel
    sepia[:, :, 1] *= 204  # Green channel
    sepia[:, :, 2] *= 255  # Red channel
    
    # Multiply the grayscale image with the sepia color
    sepia[:, :, 0] *= normalized_gray
    sepia[:, :, 1] *= normalized_gray
    sepia[:, :, 2] *= normalized_gray
    
    return np.array(sepia, np.uint8)

def apply_warm_effect(frame: np.ndarray) -> np.ndarray:
    """Applies a warm color temperature effect."""
    # Increase red, decrease blue
    float_frame = frame.astype(np.float32)
    float_frame[:, :, 2] = np.clip(float_frame[:, :, 2] * 1.2, 0, 255)
    float_frame[:, :, 0] = np.clip(float_frame[:, :, 0] * 0.8, 0, 255)
    return float_frame.astype(np.uint8)

def apply_cold_effect(frame: np.ndarray) -> np.ndarray:
    """Applies a cold color temperature effect."""
    # Increase blue, decrease red
    float_frame = frame.astype(np.float32)
    float_frame[:, :, 0] = np.clip(float_frame[:, :, 0] * 1.2, 0, 255)
    float_frame[:, :, 2] = np.clip(float_frame[:, :, 2] * 0.8, 0, 255)
    return float_frame.astype(np.uint8)

def apply_sketch_effect(bgr_image: np.ndarray) -> np.ndarray:
    """Creates a pencil sketch from a BGR image frame."""
    gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)
    # Convert the single-channel sketch back to a 3-channel BGR image for the stream
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# --- Webcam and Snapshot Functionality ---

class VideoTransformer(VideoTransformerBase):
    """A class to transform video frames from the webcam."""
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.in_frame = None
        self.out_frame = None
        # Add an attribute to hold the current effect state
        self.effect = "None"

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        in_frame = frame.to_ndarray(format="bgr24")
        
        # Use the instance attribute to apply the effect
        out_frame = apply_effect(in_frame, self.effect)

        with self.frame_lock:
            self.in_frame = in_frame
            self.out_frame = out_frame
        
        return out_frame

def webcam_and_effects():
    """Handles the webcam stream and allows applying effects and taking snapshots."""
    
    st.header("Live Webcam Feed")
    selected_effect = st.sidebar.selectbox(
        "Select Live Effect", ["None", "Grayscale", "Warm", "Cold", "Sepia", "Sketch"]
    )
    
    ctx = webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
    
    # This block is key: it updates the effect on the transformer instance
    # every time the script reruns (e.g., when the selectbox changes).
    if ctx.video_transformer:
        ctx.video_transformer.effect = selected_effect

    if ctx.video_transformer:
        if st.button("Take Snapshot"):
            with ctx.video_transformer.frame_lock:
                in_frame = ctx.video_transformer.in_frame
                out_frame = ctx.video_transformer.out_frame

            if in_frame is not None and out_frame is not None:
                st.subheader("Captured Snapshot")
                col1, col2 = st.columns(2)
                with col1:
                    st.text("Original")
                    st.image(in_frame, channels="BGR")
                with col2:
                    st.text(f"With {selected_effect} Effect")
                    st.image(out_frame, channels="BGR")
                
                # Convert the processed image for download
                result_image = Image.fromarray(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
                output_buffer = io.BytesIO()
                result_image.save(output_buffer, format="PNG")

                st.download_button(
                    label=f"Download {selected_effect} Image",
                    data=output_buffer.getvalue(),
                    file_name=f"Snapshot_{selected_effect}.png",
                    mime="image/png"
                )
            else:
                st.warning("No frames available yet. Please wait a moment.")

# --- Image Processing Functions for Uploaded Images ---

def cannize_image(pil_image: Image.Image):
    """Applies Canny Edge Detection to an image."""
    st.markdown("### Canny Edge Detection")
    
    # Convert PIL image to an OpenCV compatible format (BGR)
    rgb_image = np.array(pil_image.convert('RGB'))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(bgr_image, (11, 11), 0)

    # Sliders for Canny thresholds
    threshold1 = st.sidebar.slider("Threshold 1", 10, 200, 50, 5, key="canny_thresh1")
    threshold2 = st.sidebar.slider("Threshold 2", 20, 400, 150, 5, key="canny_thresh2")

    canny_edges = cv2.Canny(blurred_img, threshold1, threshold2)
    st.image(canny_edges, caption="Canny Edges")

    # Prepare for download
    output_buffer = io.BytesIO()
    Image.fromarray(canny_edges).save(output_buffer, format="PNG")
    st.download_button(
        label="Download Canny Image",
        data=output_buffer.getvalue(),
        file_name="Canny_Image.png",
        mime="image/png"
    )

def remove_background(pil_image: Image.Image):
    """Removes the background from an image."""
    st.markdown("### Background Removal")
    
    # The 'remove' function from rembg handles PIL images directly
    image_no_bg = remove(pil_image)
    st.image(image_no_bg, caption="Background Removed")
    
    # Prepare for download
    output_buffer = io.BytesIO()
    image_no_bg.save(output_buffer, format="PNG")
    st.download_button(
        label="Download Image",
        data=output_buffer.getvalue(),
        file_name="Removed_BG_Image.png",
        mime="image/png"
    )

def cartoonize_image(pil_image: Image.Image):
    """Applies a cartoon effect to an image."""
    st.markdown("### Cartoon Effect")
    
    # Convert PIL image to OpenCV BGR format
    rgb_image = np.array(pil_image.convert('RGB'))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Get edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter to create the cartoon effect
    color = cv2.bilateralFilter(bgr_image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Display the BGR image correctly in Streamlit
    st.image(cartoon, channels="BGR", caption="Cartoonized Image")
    
    # Prepare for download
    output_buffer = io.BytesIO()
    # Convert back to RGB for saving with PIL
    result_image = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    result_image.save(output_buffer, format="PNG")
    st.download_button(
        label="Download Cartoon Image",
        data=output_buffer.getvalue(),
        file_name="Cartoon_Image.png",
        mime="image/png"
    )

def sketch_image(pil_image: Image.Image):
    """Creates a pencil sketch from an image."""
    st.markdown("### Pencil Sketch")

    # Convert PIL image to RGB numpy array, then to grayscale
    rgb_image = np.array(pil_image.convert('RGB'))
    gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray_img)
    
    # Blur the inverted image
    blurred_img = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred_img)
    
    # Create the sketch by dividing the grayscale image by the inverted-blurred image
    sketch = cv2.divide(gray_img, inverted_blurred, scale=256.0)
    st.image(sketch, caption="Pencil Sketch")
    
    # Prepare for download
    output_buffer = io.BytesIO()
    Image.fromarray(sketch).save(output_buffer, format="PNG")
    st.download_button(
        label="Download Sketch Image",
        data=output_buffer.getvalue(),
        file_name="Sketch_Image.png",
        mime="image/png"
    )

# --- Main Application Logic ---

def main():
    """The main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("🎨 Image Effects Pro")
    st.write("Upload an image, use your webcam, or try the demo to apply cool effects!")
    
    st.sidebar.title("Controls")
    choice = st.sidebar.radio("Choose your mode:", ("Browse an Image", "Webcam Feed", "Show Demo"))
    st.sidebar.write("---")

    if choice == "Browse an Image":
        st.header("Upload an Image")
        image_file = st.file_uploader("Select an image...", type=['jpg', 'png', 'jpeg'])

        if image_file:
            pil_image = Image.open(image_file)
            st.image(pil_image, caption="Original Uploaded Image", width=600)
            
            st.sidebar.write("### Apply Effects")
            st.sidebar.info("Select one or more effects to apply below. Sliders for 'Canny' will appear here.")
            
            # Use columns for a cleaner layout of checkboxes
            col1, col2 = st.sidebar.columns(2)
            with col1:
                run_canny = st.checkbox("Canny Edge")
                run_sketch = st.checkbox("Pencil Sketch")
            with col2:
                run_cartoon = st.checkbox("Cartoonize")
                run_rembg = st.checkbox("Remove BG")

            st.write("---")
            st.header("Applied Effects")
            
            if not any([run_canny, run_sketch, run_cartoon, run_rembg]):
                st.info("Select an effect from the sidebar to see the result here.")

            if run_canny:
                cannize_image(pil_image)
            if run_cartoon:
                cartoonize_image(pil_image)
            if run_sketch:
                sketch_image(pil_image)
            if run_rembg:
                remove_background(pil_image)

    elif choice == "Show Demo":
        st.header("Demo of All Effects")
        try:
            # Assumes 'MESSI.jpg' is in the same directory.
            # Replace with a URL if you want it to be more robust.
            image_path = "MESSI.jpg" 
            pil_image = Image.open(image_path)
            st.image(pil_image, caption="Demo Image: MESSI.jpg", width=600)
            
            st.write("---")
            st.info("Sliders for 'Canny' will appear in the sidebar.")
            
            # Displaying each effect
            cannize_image(pil_image)
            cartoonize_image(pil_image)
            sketch_image(pil_image)
            remove_background(pil_image)

        except FileNotFoundError:
            st.error(f"Demo image '{image_path}' not found. Please place it in the same directory as the script.")
    
    elif choice == "Webcam Feed":
        webcam_and_effects()

if __name__ == "__main__":
    main()


