from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
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

class TTSEngine:
    def __init__(self):
        self.pipelines = {}
        self.voice_models = {}  # Cache for voice-specific models
        self.models = {}
        self.device = None
        self.cuda_available = False
        self.audio_cache = {}  # In-memory audio cache
        self.cache_dir = Path("./audio_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.setup_model()
    
    def setup_model(self):
        """Initialize the TTS model and check GPU availability"""
        # Check GPU availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = 'cuda'
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU for inference")
        
        # Check local model exists
        model_path = model_dir / "kokoro-v1_0.pth"
        voices_path = model_dir / "voices"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not voices_path.exists():
            raise FileNotFoundError(f"Voices directory not found: {voices_path}")
        
        # Set environment variables for complete offline operation
        os.environ['HF_HOME'] = str(model_dir)
        os.environ['HF_HUB_CACHE'] = str(model_dir)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(model_dir)
        os.environ['KOKORO_MODEL_PATH'] = str(model_path)
        os.environ['KOKORO_VOICES_PATH'] = str(voices_path)
        
        # Force complete offline mode - no internet access
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        print("Forced offline mode - no internet access allowed")
        
        # Initialize KModel instances using local files only
        model_path = model_dir / "kokoro-v1_0.pth"
        
        try:
            # Always create CPU model from local file
            print(f"Loading CPU model from: {model_path}")
            config_path = model_dir / "config.json"
            self.models[False] = KModel(config=str(config_path), model=str(model_path)).to('cpu').eval()
            
            # Create GPU model if available
            if self.cuda_available:
                print(f"Loading GPU model from: {model_path}")
                self.models[True] = KModel(config=str(config_path), model=str(model_path)).to('cuda').eval()
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure {model_path} exists and is accessible")
            raise
        
        print(f"Models initialized - CPU: True, GPU: {self.cuda_available}")
        self.load_cache()
        self.warm_models()
    
    def get_pipeline(self, lang_code, voice):
        """Get or create pipeline for specific language and voice"""
        pipeline_key = f"{lang_code}_{voice}"
        
        if pipeline_key not in self.pipelines:
            # Initialize pipeline without model (offline mode)
            try:
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=False  # Don't load model in pipeline - use cached models
                )
                self.pipelines[pipeline_key] = pipeline
                print(f"Pipeline for '{lang_code}' with voice '{voice}' initialized successfully (offline)")
            except Exception as e:
                print(f"Error initializing pipeline for '{lang_code}' with voice '{voice}': {e}")
                raise
        return self.pipelines[pipeline_key]
    
    def get_voice_model(self, voice, use_gpu=False):
        """Get or create cached voice model"""
        model_key = f"{voice}_{use_gpu}"
        
        if model_key not in self.voice_models:
            try:
                # Load voice-specific model
                voices_path = model_dir / "voices" / f"{voice}.pt"
                if not voices_path.exists():
                    raise FileNotFoundError(f"Voice file not found: {voices_path}")
                
                # Use the appropriate base model (GPU or CPU)
                base_model = self.models[use_gpu and self.cuda_available]
                
                # Cache the voice model
                self.voice_models[model_key] = {
                    'model': base_model,
                    'voice_path': str(voices_path)
                }
                print(f"Voice model '{voice}' cached successfully (GPU: {use_gpu and self.cuda_available})")
            except Exception as e:
                print(f"Error caching voice model '{voice}': {e}")
                raise
        
        return self.voice_models[model_key]
    
    def get_cache_key(self, amount: float, currency: str, language: str, speed: float, thx_mode: bool):
        """Generate cache key for audio"""
        key_string = f"{amount}_{currency}_{language}_{speed}_{thx_mode}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def load_cache(self):
        """Load cached audio files into memory"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        print(f"Loading {len(cache_files)} cached audio files...")
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.audio_cache[cache_file.stem] = cache_data['audio_bytes']
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")
        print(f"Loaded {len(self.audio_cache)} audio files into memory cache")
    
    def save_to_cache(self, cache_key: str, audio_bytes: bytes):
        """Save audio to both memory and disk cache"""
        self.audio_cache[cache_key] = audio_bytes
        cache_file = self.cache_dir / f"{cache_key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'audio_bytes': audio_bytes}, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def get_from_cache(self, cache_key: str):
        """Get audio from memory cache"""
        return self.audio_cache.get(cache_key)
    
    def warm_models(self):
        """Pre-warm models and pipelines to reduce first-request latency"""
        print("Warming up models...")
        try:
            # Warm up both language pipelines
            for lang_code, voice in [('b', 'af_heart'), ('z', 'zf_xiaoxiao')]:
                pipeline = self.get_pipeline(lang_code, voice)
                voice_model = self.get_voice_model(voice, self.cuda_available)
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
        """Simple Chinese number conversion"""
        if num == 0:
            return "零"
        
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        units = ["", "十", "百", "千", "万"]
        
        if num < 10:
            return digits[num]
        elif num < 100:
            tens = num // 10
            ones = num % 10
            if tens == 1:
                return "十" + (digits[ones] if ones > 0 else "")
            else:
                return digits[tens] + "十" + (digits[ones] if ones > 0 else "")
        elif num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = digits[hundreds] + "百"
            if remainder > 0:
                if remainder < 10:
                    result += "零" + digits[remainder]
                else:
                    result += self.number_to_chinese(remainder)
            return result
        else:
            # For larger numbers, use simplified approach
            return str(num)  # Fallback to digits
    
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
        """Generate speech audio for the given parameters"""
        # Check cache first
        cache_key = self.get_cache_key(amount, currency, language, speed, thx_mode)
        cached_audio = self.get_from_cache(cache_key)
        if cached_audio:
            print(f"Cache hit! Returning cached audio for {cache_key}")
            return cached_audio
        
        print(f"Cache miss. Generating new audio for {cache_key}")
        
        # Default to GPU if available, otherwise CPU
        if use_gpu is None:
            use_gpu = self.cuda_available
        try:
            # Convert amount to words based on language and currency
            if language == "EN":
                if currency == "USD":
                    amount_words = self.amount_to_words_english(amount)
                else:  # KHR
                    amount_words = self.amount_to_words_khmer(amount)
                if thx_mode == True:
                    text = f"Have received {amount_words} thanks you"
                else:
                    text = f"Have received {amount_words}"
                lang_code = 'b'  # English
                voice = 'af_heart'  # Female English voice
            
            elif language == "CH":  # Chinese
                amount_words = self.amount_to_words_chinese(amount, currency)
                if thx_mode == True:
                    text = f"已收到{amount_words} 谢谢您"
                else:
                    text = f"已收到{amount_words}"  # "Have received" in Chinese
                lang_code = 'z'  # Chinese
                voice = 'zf_xiaoxiao'  # Female Chinese voice
            
            print(f"Generating text using voice '{voice}' in language '{language}', speed: {speed}, GPU: {use_gpu}")
            
            # Get cached pipeline for this language and voice
            pipeline = self.get_pipeline(lang_code, voice)
            
            # Get cached voice model
            voice_model_info = self.get_voice_model(voice, use_gpu)
            
            # Load voice pack
            pack = pipeline.load_voice(voice)
            
            # Generate phonemes and audio
            for _, ps, _ in pipeline(text, voice=voice, speed=speed):
                ref_s = pack[len(ps)-1]
                
                try:
                    # Use the cached voice model
                    audio = voice_model_info['model'](ps, ref_s, speed)
                except Exception as e:
                    if use_gpu and self.cuda_available:
                        print(f"GPU inference failed: {e}, falling back to CPU")
                        # Fallback to CPU model
                        cpu_voice_model = self.get_voice_model(voice, False)
                        audio = cpu_voice_model['model'](ps, ref_s, speed)
                    else:
                        raise e
                # Convert to mp3 format (optimized)
                try:
                    audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio.numpy()
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, audio_np, 24000, format='mp3')
                    audio_bytes = wav_buffer.getvalue()
                    
                    # Save to cache
                    self.save_to_cache(cache_key, audio_bytes)
                    print(f"Audio cached with key: {cache_key}")
                    
                    return audio_bytes
                        
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
    amount: float = Form(..., description="Amount with up to 2 decimal places"),
    currency: Currency = Form(..., description="Currency: USD or KHR"),
    language: Language = Form(..., description="Language: EN or CH"),
    speed: float = Form(0.8, description="Speech speed (0.5-2.0, default 1.0)"),
    thx_mode: bool = Form(False, description="Use (default: False)"),
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
    
    # Generate speech
    start_time = time.time()
    audio_bytes = tts_engine.generate_speech(amount, currency, language, speed, True, thx_mode)
    generation_time = time.time() - start_time
    
    print(f"Audio generated in {generation_time:.2f}s, size: {len(audio_bytes)} bytes")
    
    # Return streaming mp3 response
    return Response(
        content=audio_bytes,
        media_type="audio/mp3",
        headers={
            "Content-Length": str(len(audio_bytes)),
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