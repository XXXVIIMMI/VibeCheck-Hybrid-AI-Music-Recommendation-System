import streamlit as st

st.set_page_config(
    page_title="VibeCheck | Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import sys
import os
import textwrap
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.ui.api_client import api_client, APIError

# Load external CSS
css_file = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error("CSS file not found!")

# Check API connectivity on startup
if 'api_connected' not in st.session_state:
    try:
        health = api_client.check_health()
        st.session_state.api_connected = True
        st.session_state.api_info = health
    except APIError as e:
        st.session_state.api_connected = False
        st.session_state.api_error = str(e)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display API connection status
if not st.session_state.api_connected:
    st.error(f"""
    ⚠️ **Cannot connect to FastAPI backend!**
    
    {st.session_state.api_error}
    
    **To fix this:**
    1. Open a terminal in the project root
    2. Run: `python -m app.api.main`
    3. Wait for the server to start on port 8000
    4. Refresh this page
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3043/3043665.png", width=100)
    st.markdown("## Navigation")
    mode = st.radio("Choose Mode", ["🎵 Recommendation", "🤖 AI Assistant"])
    
    st.markdown("---")
    st.markdown("### API Status")
    if st.session_state.api_connected:
        st.success("✅ Connected")
        if 'api_info' in st.session_state:
            info = st.session_state.api_info
            st.caption(f"Model: {info.get('model_name', 'N/A')}")
            if info.get('agent_ready'):
                st.caption("🤖 AI Agent: Ready")
            else:
                st.caption("⚠️ AI Agent: Limited (missing API key)")
    else:
        st.error("❌ Disconnected")
    
    st.markdown("---")
    st.markdown("### About")
    st.caption("Music Recommender powered by TF-IDF, Cosine Similarity, and LangChain.")

# Main Interface
if mode == "🎵 Recommendation":
    st.markdown('<h1 class="title-text">VibeCheck Results</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select a Song")
        
        # Load songs from API
        if 'song_list' not in st.session_state:
            try:
                with st.spinner("Loading songs..."):
                    st.session_state.song_list = api_client.get_all_songs()
            except APIError as e:
                st.error(f"Error loading songs: {e}")
                st.session_state.song_list = []
        
        if st.session_state.song_list:
            selected_song = st.selectbox("Search your favorite track", st.session_state.song_list)
            
            if st.button("Get Recommendations"):
                if selected_song:
                    try:
                        with st.spinner(f"Analyzing lyrics for '{selected_song}'..."):
                            recommendations = api_client.get_recommendations(selected_song, top_n=5)
                            if recommendations is None:
                                st.session_state.recommendations = None
                            else:
                                # Convert to DataFrame for display
                                st.session_state.recommendations = pd.DataFrame(recommendations)
                    except APIError as e:
                        st.error(f"Error: {e}")
                        st.session_state.recommendations = None
        else:
            st.warning("No songs available. Check API connection.")
    
    with col2:
        if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
            st.subheader("Recommended Tracks")
            rec_df = st.session_state.recommendations
            
            for index, row in rec_df.iterrows():
                st.markdown(f"""
                    <div class="song-card">
                        <div class="song-title">🎶 {row['song']}</div>
                        <div class="song-artist">🎤 {row['artist']}</div>
                    </div>
                """, unsafe_allow_html=True)
        elif 'recommendations' in st.session_state and st.session_state.recommendations is None:
            st.warning("Song not found in dataset")
        else:
            st.info("Select a song and hit recommend to see the magic!")

elif mode == "🤖 AI Assistant":
    st.markdown('<h1 class="title-text">Music Assistant</h1>', unsafe_allow_html=True)
    
    # Check if AI agent is ready
    if st.session_state.api_info.get('agent_ready'):
        # Create a container for chat messages with fixed height
        # Chat container UI (CSS handled by style.css)
        
        # Chat messages container
        chat_html = '<div class="chat-container">'
        
        if not st.session_state.chat_history:
            # Welcome message when chat is empty
            chat_html += textwrap.dedent('''
                <div class="message-wrapper assistant">
                    <div class="avatar assistant">🎵</div>
                    <div class="modern-message assistant">
                        👋 Hi! I'm your Music Assistant powered by AI. Ask me for song recommendations!
                    </div>
                </div>
            ''')
        else:
            # Display chat history
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    chat_html += textwrap.dedent(f'''
                        <div class="message-wrapper user">
                            <div class="modern-message user">{content}</div>
                            <div class="avatar user">👤</div>
                        </div>
                    ''')
                else:
                    # Format assistant message
                    # Replace newlines with <br> and handle potential markdown simple cases
                    formatted_content = content.replace('\n', '<br>')
                    # Handle bold (**text** or __text__)
                    formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
                    formatted_content = re.sub(r'__(.*?)__', r'<strong>\1</strong>', formatted_content)
                    # Handle italic (*text* or _text_)
                    formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
                    formatted_content = re.sub(r'_(.*?)_', r'<em>\1</em>', formatted_content)
                    
                    chat_html += textwrap.dedent(f'''
                        <div class="message-wrapper assistant">
                            <div class="avatar assistant">🎵</div>
                            <div class="modern-message assistant">{formatted_content}</div>
                        </div>
                    ''')
        
        # Add auto-scroll trigger (using a hidden image onerror hack as it's more reliable in st.markdown)
        chat_html += textwrap.dedent('''
            </div>
            <img src="x" style="display:none" onerror="
                var container = document.querySelector('.chat-container');
                if (container) { container.scrollTop = container.scrollHeight; }
            ">
        ''')
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Beautiful input field with custom styling
        prompt = st.chat_input(
            "Ask me for song recommendations...",
            key="chat_input"
        )
        
        if prompt:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Get bot response from API
            try:
                with st.spinner("🎵 Thinking..."):
                    response = api_client.chat_with_assistant(prompt)
                
                # Add bot message
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Chat error: {str(e)}")
    else:
        st.warning("⚠️ AI Assistant is running in limited mode (Groq API Key missing)")
        st.info("The AI assistant requires a valid Groq API key. Please add it to your .env file.")

