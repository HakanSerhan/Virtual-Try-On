++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# Virtual Try-On PoC

A Python-based virtual try-on system that takes a person photo and garment image, then outputs a composite image with the garment realistically placed on the person.

## Features

- **Pose Estimation**: MediaPipe-based body keypoint detection
- **Human Parsing**: SCHP model for body part segmentation
- **Garment Warping**: TPS (Thin-Plate Spline) warping to fit garments to body
- **Occlusion Handling**: Arms and hair rendered in front of garments
- **Auto Mask Generation**: Automatic garment background removal via rembg

## Project Structure

```
Virtual Try-on/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   └── schemas.py         # Pydantic models
├── tryon/                  # Core try-on pipeline
│   ├── pipeline.py        # Main orchestrator
│   ├── pose/              # Pose estimation
│   ├── parsing/           # Human parsing
│   ├── garment/           # Garment processing
│   ├── warp/              # Image warping
│   └── composite/         # Layer compositing
├── utils/                  # Utility functions
├── models/                 # Model weights
├── streamlit_app.py       # Demo UI
└── tests/                  # Test suite
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Streamlit Demo

```bash
streamlit run streamlit_app.py
```

### API Endpoint

**POST /tryon**

```bash
curl -X POST "http://localhost:8000/tryon" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.png" \
  -F "category=upper" \
  -F "quality_mode=fast"
```

**Parameters:**
- `person_image` (required): Person photo file
- `garment_image` (required): Garment image file
- `garment_mask` (optional): Pre-computed garment mask
- `category`: "upper" (default), reserved for future expansion
- `quality_mode`: "fast" (affine warp) or "better" (TPS warp)

**Response:** PNG image binary

## Quality Modes

- **fast**: Uses affine transformation for quick results (~1s)
- **better**: Uses TPS warping with occlusion handling (~2-3s)

## Requirements

- Python 3.10+
- CUDA-capable GPU (optional, improves performance)
- 4GB+ RAM

## License

MIT License

