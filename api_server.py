from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import soundfile as sf
import numpy as np
from pathlib import Path
import os
import warnings
import torch
import io
import tempfile
import time
import hashlib
import pickle
import requests
import urllib.request
from num2words import num2words
from kokoro import KModel, KPipeline
from enum import Enum

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

app = FastAPI(title="Kokoro TTS API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dir = Path("./kokoro_model")

# Create audio cache directory before mounting static files
audio_cache_dir = Path("./audio_cache")
audio_cache_dir.mkdir(exist_ok=True)

# Mount static files for serving cached audio
app.mount("/audio", StaticFiles(directory="audio_cache"), name="audio")

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
    
    def amount_to_words_english(self, amount):
        """Convert amount to English words for USD"""
        dollars = int(amount)
        cents = round((amount - dollars) * 100)
        
        if dollars == 0:
            dollar_text = ""
        elif dollars == 1:
            dollar_text = "one dollar"
        else:
            dollar_text = f"{num2words(dollars)} dollars"
        
        if cents == 0:
            cent_text = ""
        elif cents == 1:
            cent_text = "one cent"
        else:
            cent_text = f"{num2words(cents)} cents"
        
        if dollar_text and cent_text:
            return f"{dollar_text} and {cent_text}"
        elif dollar_text:
            return dollar_text
        else:
            return cent_text if cent_text else "zero dollars"
    
    def amount_to_words_khmer(self, amount):
        """Convert amount to words for KHR (convert to integer riels)"""
        riels = int(amount)  # Convert to integer, no decimals for KHR
        
        if riels == 0:
            return "zero riels"
        elif riels == 1:
            return "one riel"
        else:
            return f"{num2words(riels)} riels"
    
    def number_to_chinese(self, num):
        """Comprehensive Chinese number conversion for numbers up to billions"""
        if num == 0:
            return "零"
        
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        
        def convert_section(n, is_beginning=True):
            """Convert a section (0-9999) to Chinese
            is_beginning: True if this is the first/leading section of the number
            """
            if n == 0:
                return ""
            elif n < 10:
                return digits[n]
            elif n < 100:
                tens = n // 10
                ones = n % 10
                if tens == 1:
                    # Special case: if this is not the beginning section, use "一十"
                    if is_beginning:
                        return "十" + (digits[ones] if ones > 0 else "")
                    else:
                        return "一十" + (digits[ones] if ones > 0 else "")
                else:
                    return digits[tens] + "十" + (digits[ones] if ones > 0 else "")
            elif n < 1000:
                hundreds = n // 100
                remainder = n % 100
                result = digits[hundreds] + "百"
                if remainder == 0:
                    return result
                elif remainder < 10:
                    return result + "零" + digits[remainder]
                else:
                    # When we have remainder in tens, it's not the beginning anymore
                    return result + convert_section(remainder, False)
            else:  # n < 10000
                thousands = n // 1000
                remainder = n % 1000
                result = digits[thousands] + "千"
                if remainder == 0:
                    return result
                elif remainder < 100:
                    if remainder < 10:
                        return result + "零" + convert_section(remainder, False)
                    else:
                        # Special handling for 10-99 after thousands
                        return result + "零" + convert_section(remainder, False)
                else:
                    return result + convert_section(remainder, False)
        
        # Handle different ranges
        if num < 10000:
            return convert_section(num, True)
        elif num < 100000000:  # Less than 1 yi (100 million)
            wan = num // 10000
            remainder = num % 10000
            result = convert_section(wan, True) + "万"
            if remainder == 0:
                return result
            elif remainder < 1000:
                if remainder < 100:
                    if remainder < 10:
                        return result + "零" + convert_section(remainder, False)
                    else:
                        # Special case: numbers like 50010, 30010 - need "零一十"
                        return result + "零" + convert_section(remainder, False)
                else:
                    return result + "零" + convert_section(remainder, False)
            else:
                return result + convert_section(remainder, False)
        else:  # 1 yi or more
            yi = num // 100000000
            remainder = num % 100000000
            result = convert_section(yi, True) + "亿"
            if remainder == 0:
                return result
            elif remainder < 10000000:  # Less than 1000 wan
                if remainder < 10000:
                    if remainder < 1000:
                        if remainder < 100:
                            if remainder < 10:
                                return result + "零" + convert_section(remainder, False)
                            else:
                                return result + "零" + convert_section(remainder, False)
                        else:
                            return result + "零" + convert_section(remainder, False)
                    else:
                        return result + "零" + convert_section(remainder, False)
                else:
                    wan_part = remainder // 10000
                    final_remainder = remainder % 10000
                    wan_result = convert_section(wan_part, False) + "万"
                    if final_remainder == 0:
                        return result + wan_result
                    elif final_remainder < 1000:
                        if final_remainder < 100:
                            if final_remainder < 10:
                                return result + wan_result + "零" + convert_section(final_remainder, False)
                            else:
                                return result + wan_result + "零" + convert_section(final_remainder, False)
                        else:
                            return result + wan_result + "零" + convert_section(final_remainder, False)
                    else:
                        return result + wan_result + convert_section(final_remainder, False)
            else:
                return result + self.number_to_chinese(remainder)
    
    def amount_to_words_chinese(self, amount, currency):
        """Convert amount to Chinese words"""
        try:
            if currency == "USD":
                dollars = int(amount)
                cents = round((amount - dollars) * 100)
                
                if dollars == 0:
                    dollar_text = ""
                else:
                    dollar_text = f"{self.number_to_chinese(dollars)}美元"
                
                if cents == 0:
                    cent_text = ""
                else:
                    cent_text = f"{self.number_to_chinese(cents)}美分"
                
                if dollar_text and cent_text:
                    return f"{dollar_text}{cent_text}"
                elif dollar_text:
                    return dollar_text
                else:
                    return cent_text if cent_text else "零美元"
            
            else:  # KHR
                riels = int(amount)
                if riels == 0:
                    return "零瑞尔"
                else:
                    return f"{self.number_to_chinese(riels)}瑞尔"
        except Exception as e:
            print(f"Error in Chinese conversion: {e}")
            # Fallback to simple format
            if currency == "USD":
                return f"{amount}美元"
            else:
                return f"{int(amount)}瑞尔"
    
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
                    amount_words = self.amount_to_words_english(amount)
                else:  # KHR
                    amount_words = self.amount_to_words_khmer(amount)
                if thx_mode == True:
                    text = f"{amount_words} received. Thanks you"
                else:
                    text = f"{amount_words} received"
                lang_code = 'a'  # English
                voice = 'af_heart'  # Female English voice
            
            elif language == "CH":  # Chinese
                amount_words = self.amount_to_words_chinese(amount, currency)
                if thx_mode == True:
                    text = f"已收到{amount_words}. 谢谢"
                else:
                    text = f"已收到{amount_words}"  # "Have received" in Chinese
                lang_code = 'z'  # Chinese
                voice = 'zf_xiaoni'  # Female Chinese voice
            
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
        
# Initialize TTS engine
tts_engine = TTSEngine()

class Currency(str, Enum):
    USD = "USD"
    KHR = "KHR"

class Language(str, Enum):
    EN = "EN"
    CH = "CH"

@app.post("/voicegenerate")
async def voice_generate(
    amount: float,
    currency: Currency,
    language: Language,
    speed: float = 0.8,
    thx_mode: bool = False,
):
    """
    Generate voice audio for received amount
    
    Parameters:
    - amount: Float with max 2 decimal places (e.g., 100.00, 2882283.50)
    - currency: "USD" or "KHR"
    - language: "EN" or "CH"
    - speed: Speech speed (0.5-2.0, default 1.0)
    - use_gpu: Force GPU/CPU usage (default: auto-detect)
    
    Returns streaming mp3 audio response
    """
    
    # Validate inputs
    if currency not in ["USD", "KHR"]:
        raise HTTPException(status_code=400, detail="Currency must be 'USD' or 'KHR'")
    
    if language not in ["EN", "CH"]:
        raise HTTPException(status_code=400, detail="Language must be 'EN' or 'CH'")
    
    # Validate amount format (max 2 decimal places)
    if round(amount, 2) != amount:
        raise HTTPException(status_code=400, detail="Amount must have maximum 2 decimal places")
    
    if amount < 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    
    # Validate speed parameter
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")
    
    # PRINT REQUEST RECIVED
    print(f"Received request: amount={amount}, currency={currency}, language={language}, speed={speed}, use_gpu={True}") 
    
    # Generate speech (returns file path for cached, bytes for new)
    start_time = time.time()
    result = tts_engine.generate_speech(amount, currency, language, speed, True, thx_mode)
    generation_time = time.time() - start_time
    
    # Check if result is a file path (cached) or bytes (newly generated)
    if isinstance(result, Path):
        print(f"Serving cached file in {generation_time:.2f}s: {result}")
        return FileResponse(
            path=str(result),
            media_type="audio/mp3",
            filename="voice.mp3"
        )
    else:
        # This shouldn't happen with the new implementation, but keeping as fallback
        print(f"Audio generated in {generation_time:.2f}s, size: {len(result)} bytes")
        return Response(
            content=result,
            media_type="audio/mp3",
            headers={
                "Content-Length": str(len(result)),
                "Content-Disposition": "attachment; filename=voice.mp3"
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": tts_engine.device,
        "model_loaded": len(tts_engine.pipelines) > 0
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1  # Single worker to avoid GPU memory issues
    )