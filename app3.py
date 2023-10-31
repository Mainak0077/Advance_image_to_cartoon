import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from rembg import remove
import av
import random
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from typing import Union
import threading
import tempfile

captured_images = []

processed_image=None
# camera_feed=None

def apply_effect(frame,effect_type):
    if effect_type=="Grayscale":
        grayscale_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(grayscale_frame,cv2.COLOR_GRAY2BGR)
    
    elif effect_type=="Warm":
        return apply_Warm_effect(frame)
    
    elif effect_type=="Cold":
        return apply_Cold_effect(frame)
    
    elif effect_type=="Sepia":
        return apply_sepia_effect(frame)
    else:
        return frame
    
    
    

def apply_sepia_effect(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32) / 255
    # Solid color
    sepia = np.ones(src_image.shape)
    sepia[:, :, 0] *= 153  # B
    sepia[:, :, 1] *= 204  # G
    sepia[:, :, 2] *= 255  # R
    # Hadamard
    sepia[:, :, 0] *= normalized_gray  # B
    sepia[:, :, 1] *= normalized_gray  # G
    sepia[:, :, 2] *= normalized_gray  # R
    return np.array(sepia, np.uint8)



def apply_Cold_effect(frame):
    frame=frame.astype(np.float32)
    frame[:,:,0]-=10
    frame[:,:,1]-=25
    
    frame[frame<0]=0
    frame[frame>255]=255
    return frame.astype(np.uint8)
    
    

def apply_Warm_effect(frame):
    frame=frame.astype(np.float32)
    frame[:,:,0]+=10
    frame[:,:,1]+=25
    frame[frame>255]=255
    return frame.astype(np.uint8)

  
def webcam():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock
        in_frame: Union[np.ndarray, None]
        out_frame: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_frame = None
            self.out_frame = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_frame = frame.to_ndarray(format="bgr24")

            # Apply the gray scale effect to the frame
            out_frame = apply_effect(in_frame,selected_effect)

            with self.frame_lock:
                self.in_frame = in_frame
                self.out_frame = out_frame

            return out_frame
    
    selected_effect=st.sidebar.selectbox("Select Effect",["None","Grayscale","Warm","Cold","Sepia"])

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)
   
 

    if ctx.video_transformer:
        if st.button("Snapshot"):
            with ctx.video_transformer.frame_lock:
                in_frame = ctx.video_transformer.in_frame
                out_frame = ctx.video_transformer.out_frame

            if in_frame is not None and out_frame is not None:
                st.write("Original Frame:")
                st.image(in_frame, channels="BGR")
                st.write(f"{selected_effect} Effect Frame:")
                st.image(out_frame, channels="BGR")
                
                
                
                output_buffer = io.BytesIO()
                Image.fromarray(out_frame).save(output_buffer, format="PNG")

                # Create a download button for the output image
                st.download_button(
                    label="Download",
                    data=output_buffer.getvalue(),
                    file_name="Snapshot_Image.png",
                    mime="image/png"
                )
                
              

      
            else:
                st.warning("No frames available yet")
    


def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)

    threshold1 = st.sidebar.slider("Threshold 1", 10, 110, 50, 20)
    threshold2 = st.sidebar.slider("Threshold 2", 80, 200, 150, 20)

    canny = cv2.Canny(img, threshold1, threshold2)

    st.markdown("### Canonized Image")
    st.image(canny, width=500)

    output_buffer = io.BytesIO()
    Image.fromarray(canny).save(output_buffer, format="PNG")

    st.download_button(
        label="Download",
        data=output_buffer.getvalue(),
        file_name="Cannized_Image.png",
        mime="image/png"
    )

def removebg(our_image):
    new_image = np.array(our_image.convert('RGB'))
    image_rem = remove(new_image)

    st.markdown("### Remove Background")
    st.image(image_rem)

    output_buffer = io.BytesIO()
    Image.fromarray(image_rem).save(output_buffer, format="PNG")

    st.download_button(
        label="Download",
        data=output_buffer.getvalue(),
        file_name="Cannized_Image.png",
        mime="image/png",
        key="Removebg_background_button"
    )

def cartoonize_image(our_image):
    new_img = our_image
    new_img = np.array(new_img)
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    st.markdown("#### Cartoonized Image")
    st.image(cartoon, width=500)

    output_buffer = io.BytesIO()
    Image.fromarray(cartoon).save(output_buffer, format="PNG")

    st.download_button(
        label="Download",
        data=output_buffer.getvalue(),
        file_name="Cartoon_Image.png",
        mime="image/png"
    )

def sketch_image(our_image):
    img = our_image
    img = np.array(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)

    st.markdown("#### Sketch Image")
    st.image(sketch, width=500)

    output_buffer = io.BytesIO()
    Image.fromarray(sketch).save(output_buffer, format="PNG")

    st.download_button(
        label="Download",
        data=output_buffer.getvalue(),
        file_name="Sketch_Image.png",
        mime="image/png"
    )

def main():
    st.title("Image Effects App")
    choice = st.radio("", ("Show Demo", "Browse an Image","Webcam"))
    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    if choice == "Browse an Image":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.image(our_image, width=500)

            with col1:
                cannize_checkbox = st.checkbox("Cannize")
            with col2:
                cartoonize_checkbox = st.checkbox("Cartoonize")
            with col3:
                sketch_checkbox = st.checkbox("Sketch")
            with col4:
                Removebg_checkbox = st.checkbox("RemoveBG")

            if cannize_checkbox:               
                cannize_image(our_image)
            
            if cartoonize_checkbox:
                cartoonize_image(our_image)
            
            if sketch_checkbox:
                sketch_image(our_image)

            if Removebg_checkbox:
                removebg(our_image)
    
    if choice == "Show Demo":
        our_image = Image.open("MESSI.jpg")
        st.image("Messi.jpg", width=500)
        cannize_image(our_image)
        cartoonize_image(our_image)
        sketch_image(our_image)
        removebg(our_image)
        
    if choice == "Webcam":
        webcam()
     
                
if __name__ == "__main__":
    main()

               
