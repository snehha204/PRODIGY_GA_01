import streamlit as st
from diffusers import StableDiffusionPipeline

# Page Configuration
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🌷",
    layout="centered"
)
st.markdown(
    """
    <style>
    /* Style input boxes and select boxes */
    div[data-baseweb="select"] > div,
    input {
        background-color: #FFE4EC !important;
        border-radius: 10px !important;
        border: 1px solid #E91E63 !important;
        color: #2E2E2E !important;
    }

    /* Dropdown options */
    ul {
        background-color: #FFF1F6 !important;
    }

    /* Focus effect */
    div[data-baseweb="select"] > div:focus-within,
    input:focus {
        border-color: #E91E63 !important;
        box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🌷 AI Text to Image Generator")
st.write("Generate images from text using a pre-trained Stable Diffusion model ✨")

st.markdown("### 💡 Choose what you want to generate")

category = st.selectbox(
    "Select a category",
    [
        "Write my own prompt",
        "Nature",
        "City & Architecture",
        "Fantasy & Sci-Fi",
        "Animals",
        "Art & Creative"
    ]
)

suggestions = {
    "Nature": [
        "A peaceful village near a river at sunset",
        "A mountain landscape with clouds and pine trees",
        "A calm beach during golden hour"
    ],
    "City & Architecture": [
        "A futuristic city at night with neon lights",
        "An old European street with cafes",
        "A modern skyscraper skyline"
    ],
    "Fantasy & Sci-Fi": [
        "A dragon flying over a medieval castle",
        "A sci-fi robot in a cyberpunk city",
        "A fantasy forest with glowing plants"
    ],
    "Animals": [
        "A cat wearing sunglasses",
        "A majestic lion in the savannah",
        "A cute puppy illustration"
    ],
    "Art & Creative": [
        "A watercolor painting of flowers",
        "A pencil sketch of a human face",
        "An abstract colorful artwork"
    ]
}

# User Input
prompt = st.text_input(
    "Enter image description 👇🏻",
    placeholder="e.g. A futuristic city at night"
)

# Load Model 
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo"
    )
    pipe = pipe.to("cpu")   # Optimized for Mac
    return pipe

pipe = load_model()

# Generate Image
if st.button("🚀 Generate Image"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a text prompt.")
    else:
        with st.spinner("Generating image... Please wait ⏳"):

            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]

            
            image.save("generated_image.png")

            
            st.image(image, caption="Generated Image", use_column_width=True)

            # Download button 
            with open("generated_image.png", "rb") as file:
                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name="ai_generated_image.png",
                    mime="image/png"
                )

            st.success("✅ Image generated successfully!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Generative AI")