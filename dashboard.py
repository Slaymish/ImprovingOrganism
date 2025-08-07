#!/usr/bin/env python3
"""
Real-time Dashboard for ImprovingOrganism
Visualizes system progress, conversations, and learning metrics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
from datetime import datetime, timedelta
import time
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="ImprovingOrganism Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
DB_PATH = "data/memory.db"

@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_system_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_memory_stats():
    """Get statistics from the memory database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check if table exists
        table_check = pd.read_sql_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='memory'
        """, conn)
        
        if table_check.empty:
            conn.close()
            return {
                "entry_counts": pd.DataFrame(),
                "recent_activity": pd.DataFrame(),
                "score_trends": pd.DataFrame(),
                "session_stats": pd.DataFrame()
            }
        
        # Total entries by type
        entry_counts = pd.read_sql_query("""
            SELECT entry_type, COUNT(*) as count 
            FROM memory 
            GROUP BY entry_type
        """, conn)
        
        # Recent activity (last 24 hours)
        recent_activity = pd.read_sql_query("""
            SELECT entry_type, COUNT(*) as count,
                   datetime(timestamp) as timestamp
            FROM memory 
            WHERE timestamp > datetime('now', '-1 day')
            GROUP BY entry_type, date(timestamp)
            ORDER BY timestamp DESC
        """, conn)
        
        # Score trends
        score_trends = pd.read_sql_query("""
            SELECT timestamp, score, entry_type
            FROM memory 
            WHERE score IS NOT NULL 
            AND timestamp > datetime('now', '-7 days')
            ORDER BY timestamp
        """, conn)
        
        # Session analysis
        session_stats = pd.read_sql_query("""
            SELECT session_id, COUNT(*) as interactions,
                   AVG(score) as avg_score,
                   MIN(timestamp) as first_interaction,
                   MAX(timestamp) as last_interaction
            FROM memory 
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY last_interaction DESC
            LIMIT 20
        """, conn)
        
        conn.close()
        
        return {
            "entry_counts": entry_counts,
            "recent_activity": recent_activity,
            "score_trends": score_trends,
            "session_stats": session_stats
        }
    except Exception as e:
        st.error(f"Database error: {e}")
        return {
            "entry_counts": pd.DataFrame(),
            "recent_activity": pd.DataFrame(),
            "score_trends": pd.DataFrame(),
            "session_stats": pd.DataFrame()
        }

def get_recent_conversations():
    """Get recent conversations from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conversations = pd.read_sql_query("""
            SELECT timestamp, entry_type, content, score, session_id
            FROM memory 
            WHERE entry_type IN ('prompt', 'output', 'feedback')
            ORDER BY timestamp DESC
            LIMIT 50
        """, conn)
        conn.close()
        return conversations
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return pd.DataFrame()

def generate_test_response(prompt):
    """Generate a response for testing"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={"text": prompt},
            timeout=60  # Increased timeout to 60 seconds
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def submit_feedback(prompt, output, score, comment, session_id):
    """Submit feedback to the system"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "prompt": prompt,
                "output": output,
                "score": score,
                "comment": comment,
                "session_id": session_id
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Feedback submission failed: {e}")
        return False

# Main Dashboard
def main():
    st.title("🧠 ImprovingOrganism Dashboard")
    st.markdown("Real-time monitoring and interaction with your AI learning system")
    
    # Initialize session state
    if 'current_response' not in st.session_state:
        st.session_state.current_response = {}
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # System status
        system_online = get_system_status()
        if system_online:
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
            st.stop()
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        
        # Manual refresh
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
    
    # Get data
    stats = get_memory_stats()
    if not stats:
        st.error("Could not load system data")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_entries = stats["entry_counts"]["count"].sum() if not stats["entry_counts"].empty else 0
        st.metric("Total Entries", total_entries)
    
    with col2:
        feedback_count = stats["entry_counts"][stats["entry_counts"]["entry_type"] == "feedback"]["count"].sum() if not stats["entry_counts"].empty else 0
        st.metric("Feedback Received", feedback_count)
    
    with col3:
        avg_score = stats["score_trends"]["score"].mean() if not stats["score_trends"].empty else 0
        st.metric("Average Score", f"{avg_score:.2f}" if avg_score else "N/A")
    
    with col4:
        recent_sessions = len(stats["session_stats"]) if not stats["session_stats"].empty else 0
        st.metric("Active Sessions", recent_sessions)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Entry Distribution")
        if not stats["entry_counts"].empty:
            fig = px.pie(stats["entry_counts"], values="count", names="entry_type", 
                        title="Distribution of Entry Types")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available yet")
    
    with col2:
        st.subheader("📈 Score Trends")
        if not stats["score_trends"].empty:
            stats["score_trends"]["timestamp"] = pd.to_datetime(stats["score_trends"]["timestamp"])
            fig = px.line(stats["score_trends"], x="timestamp", y="score", 
                         color="entry_type", title="Score Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No score data available yet")
    
    # Interactive testing section
    st.header("🎮 Interactive Testing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_prompt = st.text_area("Enter a prompt to test the system:", 
                                  placeholder="e.g., 'Explain machine learning'",
                                  height=100)
        
        if st.button("🚀 Generate Response", type="primary"):
            if test_prompt:
                with st.spinner("Generating response..."):
                    result = generate_test_response(test_prompt)
                
                if "error" not in result:
                    st.success("Response generated!")
                    
                    # Store in session state for rating
                    if 'current_response' not in st.session_state:
                        st.session_state.current_response = {}
                    
                    st.session_state.current_response = {
                        'prompt': test_prompt,
                        'output': result.get("output", ""),
                        'session_id': result.get("session_id", "")
                    }
                else:
                    st.error(f"Generation failed: {result['error']}")
            else:
                st.warning("Please enter a prompt")
        
        # Show current response if available
        if 'current_response' in st.session_state and st.session_state.current_response:
            st.subheader("Generated Response:")
            st.text_area("Response:", 
                        value=st.session_state.current_response['output'], 
                        height=150, 
                        disabled=True)
            
            # Feedback section with unique keys
            st.subheader("Rate this response:")
            with st.form("feedback_form"):
                score = st.slider("Score (1-5)", 1.0, 5.0, 3.0, 0.1, key="feedback_score")
                comment = st.text_input("Optional comment:", key="feedback_comment")
                submit_feedback_btn = st.form_submit_button("Submit Feedback")
                
                if submit_feedback_btn:
                    if submit_feedback(
                        st.session_state.current_response['prompt'], 
                        st.session_state.current_response['output'], 
                        score, 
                        comment, 
                        st.session_state.current_response['session_id']
                    ):
                        st.success("Feedback submitted! The system will learn from this.")
                        # Clear the current response after feedback
                        st.session_state.current_response = {}
                        st.rerun()
                    else:
                        st.error("Failed to submit feedback")
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        - Try different types of questions
        - Rate responses honestly (1-5)
        - Add comments for detailed feedback
        - The system learns from your ratings
        - Check back later to see improvements
        """)
    
    # Previous responses rating section
    st.header("📝 Rate Previous Responses")
    
    conversations = get_recent_conversations()
    if not conversations.empty:
        # Group conversations by session for easy rating
        sessions_with_responses = conversations[
            conversations['entry_type'].isin(['prompt', 'output'])
        ].groupby('session_id')
        
        if len(sessions_with_responses) > 0:
            st.subheader("Select a conversation to rate:")
            
            # Create a selectbox for choosing conversations
            session_options = []
            session_mapping = {}
            
            for session_id, session_data in sessions_with_responses:
                if session_id and len(session_data) >= 2:  # Must have both prompt and output
                    prompts = session_data[session_data['entry_type'] == 'prompt']
                    outputs = session_data[session_data['entry_type'] == 'output']
                    
                    if not prompts.empty and not outputs.empty:
                        prompt_text = prompts.iloc[0]['content'][:50] + "..."
                        timestamp = pd.to_datetime(prompts.iloc[0]['timestamp']).strftime("%Y-%m-%d %H:%M")
                        display_text = f"{timestamp}: {prompt_text}"
                        session_options.append(display_text)
                        session_mapping[display_text] = {
                            'session_id': session_id,
                            'prompt': prompts.iloc[0]['content'],
                            'output': outputs.iloc[0]['content'],
                            'timestamp': timestamp
                        }
            
            if session_options:
                selected_session = st.selectbox(
                    "Choose a conversation:",
                    options=session_options,
                    key="session_selector"
                )
                
                if selected_session and selected_session in session_mapping:
                    session_data = session_mapping[selected_session]
                    
                    # Display the conversation
                    with st.container():
                        st.markdown(f"**� Your Question ({session_data['timestamp']}):**")
                        st.markdown(f"_{session_data['prompt']}_")
                        
                        st.markdown("**🤖 AI Response:**")
                        st.markdown(session_data['output'])
                        
                        # Check if this conversation already has feedback
                        existing_feedback = conversations[
                            (conversations['session_id'] == session_data['session_id']) & 
                            (conversations['entry_type'] == 'feedback')
                        ]
                        
                        if not existing_feedback.empty:
                            existing_score = existing_feedback.iloc[0]['score']
                            st.info(f"✅ Already rated: {existing_score}/5")
                        else:
                            # Rating form for previous responses
                            with st.form(f"rate_previous_{session_data['session_id']}"):
                                st.markdown("**Rate this conversation:**")
                                prev_score = st.slider(
                                    "Score (1-5)", 
                                    1.0, 5.0, 3.0, 0.1, 
                                    key=f"prev_score_{session_data['session_id']}"
                                )
                                prev_comment = st.text_input(
                                    "Optional comment:", 
                                    key=f"prev_comment_{session_data['session_id']}"
                                )
                                submit_prev_feedback = st.form_submit_button("Submit Rating")
                                
                                if submit_prev_feedback:
                                    if submit_feedback(
                                        session_data['prompt'],
                                        session_data['output'],
                                        prev_score,
                                        prev_comment,
                                        session_data['session_id']
                                    ):
                                        st.success("Rating submitted! Thank you for your feedback.")
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error("Failed to submit rating")
            else:
                st.info("No conversations available to rate yet. Generate some responses first!")
        else:
            st.info("No complete conversations found. Try generating some responses first!")
    else:
        st.info("No conversation history yet. Use the interactive testing above to get started!")
    
    # Recent conversations
    st.header("💬 Recent Activity")
    conversations = get_recent_conversations()
    
    if not conversations.empty:
        # Group conversations by session
        for session_id in conversations["session_id"].dropna().unique()[:5]:
            with st.expander(f"Session: {session_id[:8]}..."):
                session_data = conversations[conversations["session_id"] == session_id]
                for _, row in session_data.iterrows():
                    timestamp = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    
                    if row["entry_type"] == "prompt":
                        st.markdown(f"**🙋 User ({timestamp}):** {row['content']}")
                    elif row["entry_type"] == "output":
                        st.markdown(f"**🤖 AI:** {row['content']}")
                    elif row["entry_type"] == "feedback":
                        score_text = f"(Score: {row['score']})" if row['score'] else ""
                        st.markdown(f"**⭐ Feedback:** {row['content']} {score_text}")
    else:
        st.info("No conversations yet. Try the interactive testing above!")
    
    # Session statistics
    if not stats["session_stats"].empty:
        st.header("📊 Session Statistics")
        st.dataframe(
            stats["session_stats"].head(10),
            column_config={
                "session_id": "Session ID",
                "interactions": st.column_config.NumberColumn("Interactions"),
                "avg_score": st.column_config.NumberColumn("Avg Score", format="%.2f"),
                "first_interaction": "First",
                "last_interaction": "Last"
            },
            hide_index=True
        )

if __name__ == "__main__":
    main()
