"""
Virtual Try-On Demo - Supports both Cloud AI and Local CUDA modes.
"""

import io
import logging
import torch

import streamlit as st
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide",
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    h1, h2, h3 { 
        font-family: 'Outfit', sans-serif !important; 
        color: #fff !important; 
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .subtitle {
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .mode-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0.5rem 0;
    }
    
    .gpu-info {
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "cuda_version": torch.version.cuda,
        }
    return {"available": False}


def image_to_array(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file)
    return np.array(image.convert('RGB'))


def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


@st.cache_resource
def load_cuda_tryon():
    from tryon.cuda_tryon import CUDATryOn
    return CUDATryOn()


@st.cache_resource  
def load_hf_tryon():
    from tryon.hf_tryon import HuggingFaceTryOn
    return HuggingFaceTryOn()


def main():
    st.markdown('<h1 class="main-title">AI Virtual Try-On</h1>', unsafe_allow_html=True)
    
    # Check GPU
    gpu_info = get_gpu_info()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Mod Secimi")
        
        if gpu_info["available"]:
            st.markdown(f"""
            <div class="gpu-info">
                <b>GPU Bulundu!</b><br>
                {gpu_info['name']}<br>
                VRAM: {gpu_info['memory']}<br>
                CUDA: {gpu_info['cuda_version']}
            </div>
            """, unsafe_allow_html=True)
            
            mode = st.radio(
                "Calisma Modu",
                options=["local_cuda", "cloud_hf"],
                format_func=lambda x: {
                    "local_cuda": "Lokal GPU (CUDA)",
                    "cloud_hf": "Cloud (HuggingFace)"
                }[x],
                index=0,
            )
        else:
            st.warning("GPU bulunamadi! Sadece Cloud modu kullanilabilir.")
            mode = "cloud_hf"
        
        st.markdown("---")
        
        st.markdown("### Ayarlar")
        
        if mode == "local_cuda":
            num_steps = st.slider("Steps", 20, 50, 30, help="Daha fazla = daha iyi kalite")
            guidance = st.slider("Guidance", 5.0, 15.0, 7.5, help="Prompt'a baglilik")
            seed = st.number_input("Seed", 0, 99999, 42)
        else:
            category = st.selectbox(
                "Kategori",
                ["upper_body", "lower_body", "dresses"],
                format_func=lambda x: {"upper_body": "Ust Giyim", "lower_body": "Alt Giyim", "dresses": "Elbise"}[x]
            )
            num_steps = st.slider("Steps", 20, 40, 30)
            seed = st.number_input("Seed", -1, 99999, 42)
        
        st.markdown("---")
        
        st.markdown("### Bilgi")
        if mode == "local_cuda":
            st.info("Lokal GPU kullaniliyor. Internet gerekmez. Ilk calistirmada model indirilir (~4GB).")
        else:
            st.info("HuggingFace Space kullaniliyor. Internet gerekli. Sunucu mesgul olabilir.")
    
    # Mode indicator
    if mode == "local_cuda":
        st.markdown('<p class="subtitle">Lokal CUDA - Offline AI</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="subtitle">Cloud HuggingFace - Online AI</p>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fotograf")
        person_file = st.file_uploader("Fotografinizi yukleyin", type=["jpg", "jpeg", "png"], key="person")
        if person_file:
            st.image(Image.open(person_file), caption="Fotograf", use_container_width=True)
    
    with col2:
        st.markdown("### Kiyafet")
        garment_file = st.file_uploader("Kiyafet yukleyin", type=["jpg", "jpeg", "png"], key="garment")
        if garment_file:
            st.image(Image.open(garment_file), caption="Kiyafet", use_container_width=True)
    
    # Process button
    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        can_process = person_file is not None and garment_file is not None
        process_btn = st.button("AI ile Dene", disabled=not can_process, use_container_width=True)
    
    # Processing
    if process_btn and can_process:
        st.markdown("---")
        st.markdown("### Sonuc")
        
        progress = st.progress(0, text="Baslatiliyor...")
        
        try:
            progress.progress(10, text="Model yukleniyor...")
            
            person_arr = image_to_array(person_file)
            garment_arr = image_to_array(garment_file)
            
            progress.progress(20, text="AI calisiyor...")
            
            if mode == "local_cuda":
                tryon = load_cuda_tryon()
                result = tryon.run(
                    person_image=person_arr,
                    garment_image=garment_arr,
                    num_steps=num_steps,
                    guidance_scale=guidance,
                    seed=seed,
                )
            else:
                tryon = load_hf_tryon()
                result = tryon.run(
                    person_image=person_arr,
                    garment_image=garment_arr,
                    category=category,
                    n_steps=num_steps,
                    seed=seed,
                )
            
            progress.progress(100, text="Tamamlandi!")
            
            # Display
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                st.image(result, caption="AI Sonucu", use_container_width=True)
                
                buf = io.BytesIO()
                array_to_pil(result).save(buf, format="PNG")
                st.download_button("Indir", buf.getvalue(), "tryon_sonuc.png", "image/png", use_container_width=True)
            
            st.success("Basarili!")
            st.balloons()
            
        except Exception as e:
            progress.empty()
            st.error(f"Hata: {e}")
            logger.exception("Try-on failed")


if __name__ == "__main__":
    main()
