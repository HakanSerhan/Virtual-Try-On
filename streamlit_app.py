"""Streamlit demo for AI Virtual Try-On using Hugging Face OOTDiffusion."""

import io
import logging

import streamlit as st
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    
    .subtitle {
        color: #a0a0c0;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2.5rem;
        font-size: 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.5);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def image_to_array(uploaded_file) -> np.ndarray:
    """Convert uploaded file to numpy array."""
    image = Image.open(uploaded_file)
    return np.array(image.convert('RGB'))


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(arr)


@st.cache_resource
def load_tryon():
    """Load the AI try-on engine."""
    from tryon.hf_tryon import HuggingFaceTryOn
    return HuggingFaceTryOn()


def main():
    # Header
    st.markdown('<h1 class="main-title">AI Virtual Try-On</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Leffa AI - Ucretsiz - API gerektirmez</p>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <b>Leffa AI Virtual Try-On</b><br>
        <small>Hugging Face uzerinde ucretsiz. Islem 30-90 saniye surebilir.</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Ayarlar")
        
        category = st.selectbox(
            "Kiyafet Kategorisi",
            options=["upper_body", "lower_body", "dresses"],
            index=0,
            format_func=lambda x: {"upper_body": "Ust Giyim", "lower_body": "Alt Giyim", "dresses": "Elbise"}[x],
            help="Ust giyim, alt giyim veya elbise"
        )
        
        n_steps = st.slider(
            "Kalite (Steps)",
            min_value=20,
            max_value=40,
            value=20,
            help="Daha fazla = daha iyi ama yavas"
        )
        
        seed = st.number_input(
            "Seed (-1 = rastgele)",
            min_value=-1,
            max_value=999999,
            value=-1,
        )
        
        st.markdown("---")
        
        st.markdown("### Kullanim")
        st.markdown("""
        1. Fotografinizi yukleyin
        2. Kiyafet yukleyin
        3. **AI ile Dene** tiklayin
        4. 30-90 sn bekleyin
        5. Indirin!
        """)
        
        st.markdown("---")
        
        st.markdown("### Ipuclari")
        st.markdown("""
        **Kisi fotografi:**
        - Tam boy veya yarim boy
        - Duz durun, on cephe
        - Sade arka plan
        
        **Kiyafet:**
        - Duz gorunum
        - Beyaz arka plan ideal
        - Net goruntu
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fotografiniz")
        person_file = st.file_uploader(
            "Fotografinizi yukleyin",
            type=["jpg", "jpeg", "png"],
            key="person",
        )
        
        if person_file:
            person_image = Image.open(person_file)
            st.image(person_image, caption="Yukledginiz fotograf", use_container_width=True)
    
    with col2:
        st.markdown("### Kiyafet")
        garment_file = st.file_uploader(
            "Kiyafet gorselini yukleyin",
            type=["jpg", "jpeg", "png"],
            key="garment",
        )
        
        if garment_file:
            garment_image = Image.open(garment_file)
            st.image(garment_image, caption="Kiyafet", use_container_width=True)
    
    # Process button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        can_process = person_file is not None and garment_file is not None
        process_button = st.button(
            "AI ile Dene",
            disabled=not can_process,
            use_container_width=True
        )
    
    # Processing
    if process_button and can_process:
        st.markdown("---")
        st.markdown("### Sonuc")
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            progress_text.text("AI modeline baglaniliyor...")
            progress_bar.progress(10)
            
            # Load try-on engine
            tryon = load_tryon()
            
            progress_text.text("Gorseller hazirlaniyor...")
            progress_bar.progress(20)
            
            # Convert images
            person_arr = image_to_array(person_file)
            garment_arr = image_to_array(garment_file)
            
            progress_text.text("AI calisiyor... (30-90 saniye bekleyin)")
            progress_bar.progress(30)
            
            # Run AI try-on
            result = tryon.run(
                person_image=person_arr,
                garment_image=garment_arr,
                category=category,
                n_steps=n_steps,
                seed=seed,
            )
            
            progress_bar.progress(100)
            progress_text.text("Tamamlandi!")
            
            # Display result
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            
            with col_r2:
                st.image(result, caption="AI Sonucu", use_container_width=True)
                
                # Download button
                result_pil = array_to_pil(result)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                
                st.download_button(
                    label="Sonucu Indir",
                    data=buf.getvalue(),
                    file_name="ai_tryon_sonuc.png",
                    mime="image/png",
                    use_container_width=True,
                )
            
            st.success("AI ile basariyla olusturuldu!")
            st.balloons()
            
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            
            error_msg = str(e)
            st.error(f"Hata: {error_msg}")
            
            st.info("Sunucu mesgul olabilir. 1-2 dakika bekleyip tekrar deneyin.")
            logger.exception("AI try-on failed")


if __name__ == "__main__":
    main()
