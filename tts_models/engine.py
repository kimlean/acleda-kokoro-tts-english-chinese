import soundfile as sf
import numpy as np
from pathlib import Path
import warnings
import torch
import io
import hashlib
import urllib.request
from kokoro import KModel, KPipeline
from fastapi import HTTPException

from utils.text_conversion import TextConverter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

model_dir = Path("./kokoro_model")

class TTSEngine:
    def __init__(self):
        self.pipelines = {}
        self.voice_models = {}  # Cache for voice-specific models
        self.models = {}
        self.device = None
        self.cuda_available = False
        # File-based cache directory for MP3 files
        self.cache_dir = Path("./audio_cache")
        self.cache_dir.mkdir(exist_ok=True)
        print(f"Audio cache directory: {self.cache_dir.absolute()}")
        self.text_converter = TextConverter()
        self.setup_model()
    
    def setup_model(self):
        """Initialize the TTS model with GPU only"""
        # Check GPU availability - required for this configuration
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            raise RuntimeError("GPU is required but not available. Please ensure CUDA is properly installed.")
        
        self.device = 'cuda'
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print("GPU-only mode enabled")
        
        # Check local model exists
        model_path = model_dir / "kokoro-v1_0.pth"
        voices_path = model_dir / "voices"
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}. Downloading...")
            self.download_model_files()
        
        if not voices_path.exists():
            print(f"Voices directory not found at {voices_path}. Downloading...")
            self.download_voice_files()
        
        # TEMPORARILY DISABLE OFFLINE MODE to allow initial setup
        # Comment out these offline mode settings for now
        # os.environ['HF_HUB_OFFLINE'] = '1'
        # os.environ['TRANSFORMERS_OFFLINE'] = '1'
        # os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        # os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        print("Offline mode temporarily disabled for model loading")
        
        # Initialize KModel instances using local files only
        try:
            config_path = model_dir / "config.json"
            # Load GPU model only
            print(f"Loading GPU model from: {model_path}")
            self.models[True] = KModel(config=str(config_path), model=str(model_path)).to('cuda').eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure {model_path} exists and is accessible")
            raise
        
        print(f"GPU model initialized successfully")
        self.load_cache()
        self.warm_models()
    
    def download_model_files(self):
        """Download model files from Hugging Face"""
        base_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main"
        model_dir.mkdir(exist_ok=True)
        
        # Download main model file
        model_url = f"{base_url}/kokoro-v1_0.pth"
        model_path = model_dir / "kokoro-v1_0.pth"
        
        print(f"Downloading model from {model_url}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
        
        # Download config file
        config_url = f"{base_url}/config.json"
        config_path = model_dir / "config.json"
        
        print(f"Downloading config from {config_url}...")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            print(f"Config downloaded successfully to {config_path}")
        except Exception as e:
            print(f"Failed to download config: {e}")
            raise
    
    def download_voice_files(self):
        """Download voice files from Hugging Face"""
        base_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices"
        voices_dir = model_dir / "voices"
        voices_dir.mkdir(exist_ok=True)
        
        # List of voice files to download
        voice_files = [
            "af_heart.pt",  # English female voice
            "zf_xiaoni.pt"  # Chinese female voice
        ]
        
        for voice_file in voice_files:
            voice_url = f"{base_url}/{voice_file}"
            voice_path = voices_dir / voice_file
            
            print(f"Downloading voice {voice_file} from {voice_url}...")
            try:
                urllib.request.urlretrieve(voice_url, voice_path)
                print(f"Voice {voice_file} downloaded successfully to {voice_path}")
            except Exception as e:
                print(f"Failed to download voice {voice_file}: {e}")
                raise
            
    def get_pipeline(self, lang_code, voice):
        """Get or create pipeline for specific language and voice"""
        pipeline_key = f"{lang_code}_{voice}"
        
        if pipeline_key not in self.pipelines:
            try:
                # Pass the already loaded model instance instead of creating new one
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=self.models[True]  # Use already loaded model instance
                )
                self.pipelines[pipeline_key] = pipeline
                print(f"Pipeline for '{lang_code}' initialized with pre-loaded model")
            except Exception as e:
                print(f"Error initializing pipeline for '{lang_code}': {e}")
                raise
        return self.pipelines[pipeline_key]
    
    def get_voice_model(self, voice, use_gpu=False):
        """Get or create cached voice model with local voice loading"""
        model_key = f"{voice}_{use_gpu}"
        
        if model_key not in self.voice_models:
            try:
                # Load voice-specific model
                voices_path = model_dir / "voices" / f"{voice}.pt"
                if not voices_path.exists():
                    raise FileNotFoundError(f"Voice file not found: {voices_path}")
                
                # Load the voice pack directly from local file
                voice_pack = torch.load(voices_path, map_location='cuda' if use_gpu else 'cpu')
                
                # Use GPU model only
                base_model = self.models[True]
                
                # Cache the voice model with the loaded pack
                self.voice_models[model_key] = {
                    'model': base_model,
                    'voice_pack': voice_pack,
                    'voice_path': str(voices_path)
                }
                print(f"Voice model '{voice}' loaded from local file and cached successfully (GPU: True)")
            except Exception as e:
                print(f"Error loading local voice model '{voice}': {e}")
                raise
        
        return self.voice_models[model_key]
    
    def get_cache_key(self, amount: float, currency: str, language: str, speed: float, thx_mode: bool):
        """Generate cache key for audio"""
        key_string = f"{amount}_{currency}_{language}_{speed}_{thx_mode}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def load_cache(self):
        """Initialize file-based cache directory"""
        mp3_files = list(self.cache_dir.glob("*.mp3"))
        print(f"Found {len(mp3_files)} cached MP3 files in {self.cache_dir}")
    
    def save_to_cache(self, cache_key: str, audio_bytes: bytes):
        """Save audio as MP3 file to disk cache"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        try:
            with open(cache_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"Audio saved to: {cache_file}")
        except Exception as e:
            print(f"Error saving audio to cache: {e}")
    
    def get_from_cache(self, cache_key: str):
        """Check if cached MP3 file exists and return file path"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            return cache_file
        return None
    
    def warm_models(self):
        """Pre-warm models and pipelines to reduce first-request latency"""
        print("Warming up models...")
        try:
            # Warm up both language pipelines - use 'a' for English, 'z' for Chinese
            for lang_code, voice in [('a', 'af_heart'), ('z', 'zf_xiaoni')]:
                pipeline = self.get_pipeline(lang_code, voice)
                voice_model = self.get_voice_model(voice, True)
                print(f"Warmed up {lang_code} pipeline with voice {voice}")
            print("Model warming completed")
        except Exception as e:
            print(f"Model warming failed: {e}")
    
    def generate_speech(self, amount: float, currency: str, language: str, speed: float = 0.8, use_gpu: bool = None, thx_mode: bool = False):
        """Generate speech audio for the given parameters - using completely local approach"""
        # Check cache first
        cache_key = self.get_cache_key(amount, currency, language, speed, thx_mode)
        cached_file = self.get_from_cache(cache_key)
        if cached_file:
            print(f"Cache hit! Returning cached file: {cached_file}")
            return cached_file
        
        print(f"Cache miss. Generating new audio for {cache_key}")
        
        # GPU only mode
        use_gpu = True
        try:
            # Convert amount to words based on language and currency
            if language == "EN":
                if currency == "USD":
                    amount_words = self.text_converter.amount_to_words_english(amount)
                else:  # KHR
                    amount_words = self.text_converter.amount_to_words_khmer(amount)
                if thx_mode == True:
                    text = f"{amount_words} received. Thanks you"
                else:
                    text = f"{amount_words} received"
                lang_code = 'a'  # English
                voice = 'af_heart'  # Female English voice
            
            elif language == "CH":  # Chinese
                amount_words = self.text_converter.amount_to_words_chinese(amount, currency)
                if thx_mode == True:
                    # Old: 谢谢
                    text = f"[](/ʂoʊ˥ taʊ˥˩){amount_words}。[](/ʂjɛ˥˩ ʂjɛ˥˩)"
                else:
                    text = f"[](/ʂoʊ˥ taʊ˥˩){amount_words}"
                
                lang_code = 'z'  # Chinese
                voice = 'zf_xiaobei'  # Female Chinese voice
            
            print(f"Generating text using voice '{voice}' in language '{language}', speed: {speed}, GPU: {use_gpu}, words: {amount_words}")
            
            # Load voice pack directly from local file - bypassing pipeline voice loading
            voice_path = model_dir / "voices" / f"{voice}.pt"
            if not voice_path.exists():
                raise FileNotFoundError(f"Voice file not found: {voice_path}")
            
            voice_pack = torch.load(voice_path, map_location='cuda')
            print(f"Loaded voice pack from: {voice_path}")
            
            # Get pipeline for phoneme generation only
            pipeline = self.get_pipeline(lang_code, voice)
            
            # Generate phonemes WITH voice parameter (but we'll use our local voice pack)
            phoneme_results = list(pipeline(text, voice=voice, speed=speed))
            
            for _, ps, _ in phoneme_results:
                # Use voice pack directly
                ref_s = voice_pack[len(ps)-1] if len(ps)-1 < len(voice_pack) else voice_pack[-1]
                
                # Use GPU model for inference
                base_model = self.models[True]
                audio = base_model(ps, ref_s, speed)
                
                # Convert to mp3 format
                try:
                    audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio.numpy()
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, audio_np, 24000, format='mp3')
                    audio_bytes = wav_buffer.getvalue()
                    
                    # Save to cache
                    self.save_to_cache(cache_key, audio_bytes)
                    print(f"Audio cached with key: {cache_key}")
                    
                    # Return the cached file path
                    return self.cache_dir / f"{cache_key}.mp3"
                        
                except Exception as audio_error:
                    print(f"Audio processing failed: {audio_error}")
                    raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(audio_error)}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating speech: {e}")
            print(f"Full traceback: {error_details}")
            raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")