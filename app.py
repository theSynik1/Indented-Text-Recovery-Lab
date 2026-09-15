import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from processor import ForensicProcessor

# Set page configuration
st.set_page_config(page_title="Indented Text Recovery Lab", layout="wide")

def main():
    st.title("🔬 Indented Text Recovery Lab")
    st.markdown("""
    This tool uses forensic image processing to reveal text carved into paper (indented writing). 
    Upload a high-resolution scan or photo to begin.
    """)

    # Initialize Processor
    processor = ForensicProcessor()

    # --- Sidebar Controls ---
    st.sidebar.header("Forensic Controls")
    
    # Upload File
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Enhancement Parameters")
    
    clahe_clip = st.sidebar.slider("Contrast Intensity (CLAHE)", 0.1, 10.0, 3.0, step=0.1)
    light_angle = st.sidebar.slider("Virtual Light Angle", 0, 360, 45)
    denoise_strength = st.sidebar.slider("Noise Reduction (Bilateral)", 0, 50, 15)
    invert = st.sidebar.checkbox("Invert Colors (Negative Mode)", value=False)

    if uploaded_file is not None:
        # Load Image
        image_bytes = np.fromstring(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # --- Processing ---
        processed_img = processor.process_pipeline(
            img, 
            clahe_clip=clahe_clip, 
            light_angle=light_angle, 
            denoise_strength=denoise_strength,
            invert=invert
        )

        # --- UI Layout ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("Recovered Text View")
            st.image(processed_img, use_column_width=True, caption=f"Light Angle: {light_angle}°")

        # --- OCR and Actions ---
        st.markdown("---")
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔍 Run OCR (Text Recognition)"):
                with st.spinner("Extracting text..."):
                    extracted_text = processor.run_ocr(processed_img)
                    if extracted_text.strip():
                        st.success("Text Extracted Successfully:")
                        st.code(extracted_text)
                    else:
                        st.warning("No text detected. Try adjusting the contrast or light angle.")

        with action_col2:
            # Download button
            is_success, buffer = cv2.imencode(".png", processed_img)
            if is_success:
                st.download_button(
                    label="💾 Download Enhanced Image",
                    data=buffer.tobytes(),
                    file_name="recovered_text.png",
                    mime="image/png"
                )

    else:
        st.info("Please upload an image from the sidebar to start.")
        
        # Sample placeholder for demo feel
        st.markdown("""
        ### 💡 Tips for Best Results:
        1. **High Resolution:** 300 DPI or higher scans work best.
        2. **Consistent Lighting:** Avoid glares on the original scan.
        3. **Angle Tuning:** Slowly slide the 'Virtual Light Angle' to find where the shadows best define the pen strokes.
        4. **Negative Mode:** If the indentations are faint, Negative Mode can often make them pop out against the background.
        """)

if __name__ == "__main__":
    main()
