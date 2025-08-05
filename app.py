from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

from tts_models.routes import router as tts_router

# Create FastAPI app
app = FastAPI(title="Kokoro TTS API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create audio cache directory before mounting static files
audio_cache_dir = Path("./audio_cache")
audio_cache_dir.mkdir(exist_ok=True)

# Mount static files for serving cached audio
app.mount("/audio", StaticFiles(directory="audio_cache"), name="audio")

# Include TTS routes
app.include_router(tts_router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from tts.routes import tts_engine
    return {
        "status": "healthy",
        "device": tts_engine.device,
        "model_loaded": len(tts_engine.pipelines) > 0
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1  # Single worker to avoid GPU memory issues
    )