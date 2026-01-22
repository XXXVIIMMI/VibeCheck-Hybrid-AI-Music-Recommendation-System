"""
API Client for communicating with FastAPI backend
"""
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# FastAPI server configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds

class APIError(Exception):
    """Custom exception for API errors"""
    pass

class APIClient:
    """Client for making requests to the FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        logger.info(f"APIClient initialized with base URL: {base_url}")
    
    def check_health(self) -> Dict:
        """Check if API is healthy and available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot connect to API server. Is FastAPI running on port 8000?")
        except requests.exceptions.Timeout:
            raise APIError("API health check timed out")
        except Exception as e:
            raise APIError(f"Health check failed: {str(e)}")
    
    def get_recommendations(self, song_name: str, top_n: int = 5) -> List[Dict]:
        """
        Get song recommendations
        
        Args:
            song_name: Name of the song
            top_n: Number of recommendations to return
            
        Returns:
            List of recommendations with 'artist' and 'song' fields
            
        Raises:
            APIError: If API request fails
        """
        try:
            logger.info(f"Requesting recommendations for: {song_name}")
            response = requests.post(
                f"{self.base_url}/recommend",
                json={"song_name": song_name, "top_n": top_n},
                timeout=TIMEOUT
            )
            
            if response.status_code == 404:
                return None  # Song not found
            
            response.raise_for_status()
            recommendations = response.json()
            logger.info(f"Received {len(recommendations)} recommendations")
            return recommendations
            
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot connect to API server. Please start FastAPI first.")
        except requests.exceptions.Timeout:
            raise APIError("Request timed out. The server might be overloaded.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise APIError(f"API returned error: {e.response.status_code}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")
    
    def chat_with_assistant(self, message: str) -> str:
        """
        Chat with AI music assistant
        
        Args:
            message: User message
            
        Returns:
            Assistant's response
            
        Raises:
            APIError: If API request fails
        """
        try:
            logger.info(f"Sending chat message: {message[:50]}...")
            response = requests.post(
                f"{self.base_url}/chat",
                json={"message": message},
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response received")
            
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot connect to API server. Please start FastAPI first.")
        except requests.exceptions.Timeout:
            raise APIError("Chat request timed out. Please try again.")
        except Exception as e:
            raise APIError(f"Chat error: {str(e)}")
    
    def get_all_songs(self) -> List[str]:
        """
        Get list of all available songs
        
        Returns:
            List of song names
            
        Raises:
            APIError: If API request fails
        """
        try:
            logger.info("Fetching all songs from API")
            response = requests.get(
                f"{self.base_url}/songs",
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            songs = result.get("songs", [])
            logger.info(f"Received {len(songs)} songs")
            return songs
            
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot connect to API server. Please start FastAPI first.")
        except requests.exceptions.Timeout:
            raise APIError("Request timed out while fetching songs.")
        except Exception as e:
            raise APIError(f"Error fetching songs: {str(e)}")

# Singleton instance
api_client = APIClient()
