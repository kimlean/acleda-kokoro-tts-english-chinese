# ACLEDA TTS English-Chinese API

A FastAPI-based Text-to-Speech service supporting English and Chinese with USD/KHR currency amounts.

## Linux System Requirements

### Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libsndfile1 ffmpeg

# CentOS/RHEL/Fedora
sudo yum install -y python3 python3-pip libsndfile ffmpeg
# or for newer versions:
sudo dnf install -y python3 python3-pip libsndfile ffmpeg
```

### Audio Library Dependencies
- **libsndfile1**: Required for soundfile library (WAV audio processing)
- **ffmpeg**: Audio codec support (if needed for future extensions)

## Installation

1. **Clone and setup environment:**
```bash
git clone <your-repo-url>
cd acleda-tts-english-chinese
python3 -m venv venv
source venv/bin/activate
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup Kokoro model:**
   - Ensure `kokoro_model/` directory contains:
     - `kokoro-v1_0.pth` (model file)
     - `voices/` directory with voice files (*.pt)

## Running the API Server

### Development Mode
```bash
python api_server.py
```

### Production Mode
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

## GPU Support (Optional)

### NVIDIA GPU Setup
```bash
# Install NVIDIA drivers and CUDA toolkit
# For Ubuntu 20.04/22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## API Usage

### Endpoint: POST /voicegenerate

**Parameters:**
- `amount`: Float (max 2 decimal places)
- `currency`: "USD" or "KHR"  
- `language`: "EN" or "CH"
- `speed`: Float (0.5-2.0, default 1.0)
- `use_gpu`: Boolean (optional, auto-detect if not specified)

**Example:**
```bash
curl -X POST "http://localhost:8000/voicegenerate" \
  -F "amount=150.75" \
  -F "currency=USD" \
  -F "language=EN" \
  -F "speed=1.0" \
  --output output.wav
```

### Health Check
```bash
curl http://localhost:8000/health
```

## File Structure
```
├── api_server.py           # Main FastAPI server
├── voice_generator.py      # Voice testing utility
├── requirements.txt        # Python dependencies
├── kokoro_model/          # TTS model directory
│   ├── kokoro-v1_0.pth   # Model weights
│   └── voices/           # Voice files
└── README.md             # This file
```

## Permissions

Ensure your Linux user has read/write access to:
- `kokoro_model/` directory and contents
- Temporary file system (`/tmp`)
- Working directory for audio output

```bash
# Set proper permissions
chmod -R 755 kokoro_model/
```

## Troubleshooting

### Common Issues:

1. **libsndfile error**: Install `libsndfile1` system package
2. **CUDA not found**: Install NVIDIA drivers and CUDA toolkit
3. **Permission denied**: Check file permissions on model directory
4. **Model not found**: Verify `kokoro_model/kokoro-v1_0.pth` exists

### System Resource Usage:
- **CPU-only**: ~2GB RAM
- **GPU mode**: ~4GB GPU VRAM + 1GB RAM
- **Storage**: ~500MB for model files

## Performance

- **CPU**: ~2-5 seconds per generation
- **GPU**: ~0.5-2 seconds per generation
- **Audio format**: WAV, 24kHz sample rate
- **Supported languages**: English, Chinese (Mandarin)