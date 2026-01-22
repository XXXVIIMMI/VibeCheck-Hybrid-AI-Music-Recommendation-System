import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from app.core.recommender import recommend_songs, df
from app.core.agent import MusicAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Recommendation API",
    description="API for fetching music recommendations based on lyrics similarity.",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "http://localhost:*",      # Allow any localhost port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MusicAgent once at startup
music_agent = MusicAgent()
logger.info("MusicAgent initialized")

# Request/Response Models
class RecommendationRequest(BaseModel):
    song_name: str
    top_n: int = 5

class SongRecommendation(BaseModel):
    artist: str
    song: str
    link: Optional[str] = None
    text: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class SongsListResponse(BaseModel):
    songs: List[str]
    count: int

# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent_ready": music_agent.llm is not None,
        "model_name": music_agent.model_name
    }

@app.post("/recommend", response_model=List[SongRecommendation])
def get_recommendations(request: RecommendationRequest):
    """Get song recommendations based on lyrics similarity"""
    logger.info(f"Received recommendation request for: {request.song_name}")
    try:
        result_df = recommend_songs(request.song_name, request.top_n)
        
        if result_df is None:
            raise HTTPException(status_code=404, detail="Song not found")
        
        # Convert to list of dicts
        recommendations = result_df.to_dict(orient="records")
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):
    """Chat with AI music assistant"""
    logger.info(f"Received chat message: {request.message}")
    try:
        response = music_agent.chat(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/songs", response_model=SongsListResponse)
def get_all_songs():
    """Get list of all available songs for dropdown"""
    logger.info("Fetching all songs")
    try:
        songs = sorted(df['song'].dropna().unique().tolist())
        return SongsListResponse(songs=songs, count=len(songs))
    except Exception as e:
        logger.error(f"Error fetching songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

