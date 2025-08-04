import soundfile as sf
import numpy as np
from pathlib import Path
import os
import warnings
import torch
from huggingface_hub import snapshot_download
from kokoro import KPipeline

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

model_dir = Path("./kokoro_model")

def check_gpu_availability():
    """Check if CUDA is available and display GPU information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"GPU Available: {gpu_count} device(s)")
        print(f"Current GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("GPU: Not available, using CPU")
        return False

def check_local_model():
    """Check if local model exists and is ready"""
    model_path = model_dir / "kokoro-v1_0.pth"
    voices_path = model_dir / "voices"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.absolute()}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.absolute()}")
    
    if not voices_path.exists() or not any(voices_path.glob("*.pt")):
        raise FileNotFoundError(f"Voice files not found in: {voices_path.absolute()}")
    
    print(f"Using local model at: {model_dir.absolute()}")
    return model_path, voices_path

def test_specific_voices():
    """Test only specific high-quality voices for quick demo"""
    
    print("Testing High-Quality Voices Only")
    print("=" * 40)
    
    # Check GPU availability
    use_gpu = check_gpu_availability()
    print()
    
    # Check local model is ready
    model_path, voices_path = check_local_model()
    
    # Set environment variables to force local usage only
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(model_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(model_dir)
    os.environ['HF_HUB_CACHE'] = str(model_dir)
    os.environ['KOKORO_MODEL_PATH'] = str(model_path)
    os.environ['KOKORO_VOICES_PATH'] = str(voices_path)
    
    # Set device for PyTorch
    device = 'cuda' if use_gpu else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Best voices from each category
    best_voices = {
        'af_heart': ('b', 'English', "Have received one hundred riels"),
        'zf_xiaoxiao': ('z', 'Chinese', "你好，我是小小。我是中文女声。"),
    }
    
    output_dir = Path("best_voice_samples")
    output_dir.mkdir(exist_ok=True)
    
    for voice_name, (lang_code, language, text) in best_voices.items():
        try:
            print(f"Testing {voice_name} ({language})... ", end="")
            
            # Load from local files with GPU support
            pipeline = KPipeline(
                lang_code=lang_code,
                model=str(model_path),
                device=device
            )
            
            # Generate audio with timing
            import time
            start_time = time.time()
            generator = pipeline(text, voice=voice_name)
            
            for i, (gs, ps, audio) in enumerate(generator):
                output_file = output_dir / f"{voice_name}_{language.lower()}.wav"
                sf.write(str(output_file), audio, 24000)
            
            generation_time = time.time() - start_time
            print(f"SUCCESS ({generation_time:.2f}s)")
            
        except Exception as e:
            print(f"FAILED: {str(e)[:100]}...")
    
    print(f"\nBest samples saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    test_specific_voices()