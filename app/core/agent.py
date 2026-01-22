import os
import logging
from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .recommender import recommend_songs  # make sure this is correct

"""
Setup logging
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Load .env variables"""
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dotenv_path = os.path.join(base_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded .env from {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Using system environment variables.")

logger.info(f"MODEL_NAME from env: {os.getenv('MODEL_NAME')}")
logger.info(f"GROQ_API_KEY set: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

"""MusicAgent class"""

class MusicAgent:
    def __init__(self, api_key=None, provider=None, model_name=None):
        """Initialize the MusicAgent with Groq LLM and recommender system."""
        self.provider = provider or "Groq (Free)"
        self.model_name = model_name or os.getenv("MODEL_NAME") or "qwen/qwen3-32b"
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.llm = None
        
        """Debug logging"""
        logger.info(f"MusicAgent provider: {self.provider}")
        logger.info(f"MusicAgent model_name: {self.model_name}")
        logger.info(f"MusicAgent API key set: {'Yes' if self.api_key else 'No'}")

        if not self.api_key:
            logger.error("GROQ_API_KEY not found! LLM will not be initialized.")
        else:
            self._setup_agent()

    def _setup_agent(self):
        """Initialize the Groq LLM."""
        try:
            self.llm = ChatGroq(
                model=self.model_name,
                groq_api_key=self.api_key,
                temperature=0
            )
            logger.info(f"LLM setup successful: Groq - {self.model_name}")
        except Exception as e:
            logger.exception(f"Failed to setup Groq agent with model '{self.model_name}'")

    def _clean_thinking_tags(self, response: str) -> str:
        """Remove <think> tags from LLM response."""
        # Remove everything between <think> and </think>
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()

    def _get_recommendations(self, song_name: str) -> str:
        """Get top song recommendations."""
        try:
            logger.info(f"Fetching recommendations for: {song_name}")
            df = recommend_songs(song_name, top_n=5)
            if df is None or df.empty:
                logger.warning(f"No recommendations found for '{song_name}'")
                return "Song not found in the database. Please try another song."

            result_str = "Here are the recommended songs:\n"
            for _, row in df.iterrows():
                result_str += f"- {row['song']} by {row['artist']}\n"

            logger.info(f"Recommendations fetched:\n{result_str}")
            return result_str
        except Exception as e:
            logger.exception("Error in _get_recommendations")
            return f"Error occurred while fetching recommendations: {str(e)}"

    def _format_with_llm(self, user_input: str, recommendations: str) -> str:
        """Format recommendations using the LLM."""
        if not self.llm:
            logger.error("LLM not initialized. Returning plain text recommendations.")
            return recommendations

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful music assistant. Format the following recommendations in a friendly way."),
                ("user", f"The user asked: {user_input}\n\nRecommendations:\n{recommendations}")
            ])
            response = self.llm.invoke(prompt.format_messages())
            logger.info(f"LLM formatted response: {response.content}")
            # Clean thinking tags from response
            cleaned_response = self._clean_thinking_tags(response.content)
            return cleaned_response
        except Exception as e:
            logger.exception("LLM failed to format recommendations")
            return f"Could not format recommendations using LLM: {str(e)}"

    def chat(self, user_input: str) -> str:
        """Process user input and return recommendations or general chat response."""
        logger.info(f"User input: {user_input}")

        if not self.llm:
            logger.error("LLM not initialized. Cannot process user input.")
            return "I'm sorry, I cannot process your request because the Groq API key is missing or invalid."

        try:
            user_lower = user_input.lower()

            """Detect recommendation requests"""
            if any(word in user_lower for word in ['recommend', 'suggestion', 'similar', 'like']):
                """Case 1: Song in quotes"""
                quoted = re.findall(r'["\']([^"\']+)["\']', user_input)
                if quoted:
                    song_name = quoted[0]
                    logger.info(f"Detected quoted song name: {song_name}")
                    recommendations = self._get_recommendations(song_name)
                    return self._format_with_llm(user_input, recommendations)

                """Case 2: "similar to X" pattern"""
                similar_match = re.search(r'similar to (.+?)(?:\?|$|\.)', user_lower)
                if similar_match:
                    song_name = similar_match.group(1).strip()
                    logger.info(f"Detected 'similar to' song name: {song_name}")
                    recommendations = self._get_recommendations(song_name)
                    return self._format_with_llm(user_input, recommendations)

            """General conversation fallback"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful music assistant. You can recommend songs based on lyrics similarity. Ask the user for a song name if they want recommendations."),
                ("user", "{input}")
            ])
            response = self.llm.invoke(prompt.format_messages(input=user_input))
            logger.info(f"LLM general response: {response.content}")
            # Clean thinking tags from response
            cleaned_response = self._clean_thinking_tags(response.content)
            return cleaned_response

        except Exception as e:
            logger.exception("Exception occurred in chat()")
            return f"I encountered an error while processing your request: {str(e)}"
