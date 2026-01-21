# Virtual Try-On AI

Yapay zeka destekli sanal kıyafet deneme uygulaması. Kişi fotoğrafı ve kıyafet görseli alarak, kişinin o kıyafeti giymiş halini oluşturur.

## Özellikler

- **Lokal CUDA Modu**: NVIDIA GPU ile offline çalışır, internet gerektirmez
- **Cloud Modu**: HuggingFace üzerinden online AI kullanır
- **FastAPI Backend**: REST API desteği
- **Streamlit Demo**: Kullanımı kolay web arayüzü

## Gereksinimler

### Lokal CUDA Modu (Önerilen)
- NVIDIA GPU (en az 6GB VRAM)
- CUDA 11.8 veya 12.x
- Python 3.10+

### Cloud Modu
- Python 3.10+
- İnternet bağlantısı

## Kurulum

### 1. CUDA/GPU için (Önerilen)

```bash
# Repo'yu klonla
git clone https://github.com/HakanSerhan/Virtual-Try-On.git
cd Virtual-Try-On

# Virtual environment oluştur
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# CUDA destekli bağımlılıkları yükle
pip install -r requirements_cuda.txt
```

### 2. CPU/Cloud için

```bash
pip install -r requirements.txt
```

## Kullanım

### Streamlit Demo (Önerilen)

```bash
streamlit run streamlit_app.py
```

Tarayıcıda `http://localhost:8501` açılacak.

### FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

API dökümantasyonu: `http://localhost:8000/docs`

## Çalışma Modları

### 1. Lokal CUDA Modu
- NVIDIA GPU gerektirir
- İlk çalıştırmada model indirilir (~4GB)
- Sonraki çalıştırmalar offline çalışır
- İşlem süresi: 10-30 saniye (GPU'ya bağlı)

### 2. Cloud (HuggingFace) Modu
- GPU gerektirmez
- İnternet bağlantısı gerekli
- Ücretsiz, API key gerektirmez
- İşlem süresi: 30-90 saniye

## Proje Yapısı

```
Virtual-Try-On/
├── app/                    # FastAPI uygulaması
│   ├── main.py            # API endpoint'leri
│   └── schemas.py         # Pydantic modelleri
├── tryon/                  # Try-on pipeline
│   ├── cuda_tryon.py      # Lokal CUDA AI
│   ├── hf_tryon.py        # HuggingFace Cloud AI
│   ├── pipeline.py        # Ana pipeline
│   ├── pose/              # Poz tahmini
│   ├── parsing/           # İnsan segmentasyonu
│   ├── garment/           # Kıyafet işleme
│   ├── warp/              # Görsel dönüştürme
│   └── composite/         # Birleştirme
├── utils/                  # Yardımcı fonksiyonlar
├── streamlit_app.py       # Demo UI
├── requirements.txt       # CPU bağımlılıkları
├── requirements_cuda.txt  # CUDA bağımlılıkları
└── README.md
```

## GPU Gereksinimleri

| GPU | VRAM | Durum |
|-----|------|-------|
| RTX 4090 | 24GB | Mükemmel |
| RTX 3080 | 10GB | Çok İyi |
| RTX 3060 | 12GB | İyi |
| RTX 2060 | 6GB | Minimum |
| GTX 1660 | 6GB | Minimum |

## Sorun Giderme

### CUDA bulunamadı hatası
```bash
# PyTorch CUDA sürümünü kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# CUDA'yı yeniden yükle
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bellek yetersiz hatası
- `num_steps` değerini düşürün (20-25)
- Daha küçük görseller kullanın
- Diğer GPU uygulamalarını kapatın

### Model indirme hatası
- İnternet bağlantınızı kontrol edin
- HuggingFace'e erişimi kontrol edin
- VPN kullanmayı deneyin

## Lisans

MIT License

## Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır!
