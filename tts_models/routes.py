from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse
from pathlib import Path
import time

from models import Currency, Language
from .engine import TTSEngine

# Create router
router = APIRouter()

# Initialize TTS engine (singleton) - will be created on first import
tts_engine = None

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        tts_engine = TTSEngine()
    return tts_engine

# Initialize engine on import
tts_engine = get_tts_engine()

@router.post("/voicegenerate")
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