import streamlit as st
from diffusers import StableDiffusionPipeline

# Page Configuration

st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🌷",
    layout="centered"
)

st.title("AI Text to Image Generator")
st.write("Generate images from text using a pre-trained Stable Diffusion model")

# User Input

prompt = st.text_input(
    "Enter image description",
    placeholder="e.g. A futuristic city at night"
)

# Load Model (CPU optimized & cached)

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo"
    )
    pipe = pipe.to("cpu")   # Optimized for Mac
    return pipe

pipe = load_model()

# Generate Image

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a text prompt.")
    else:
        with st.spinner("Generating image... Please wait ⏳"):

            image = pipe(
                prompt,
                num_inference_steps=10,   # Fast
                guidance_scale=6.0,       # Balanced quality
                height=384,               # Smaller size = faster
                width=384
            ).images[0]

            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("✅ Image generated and saved as generated_image.png")
            