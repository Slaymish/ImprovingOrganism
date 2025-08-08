#!/usr/bin/env python3
"""
ImprovingOrganism Research Analytics Dashboard
Advanced metrics and evaluation interface for self-improving language models
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
import time
import numpy as np
try:
    from scipy import stats  # type: ignore
except Exception:
    class _StatsFallback:
        def skew(self, x):
            return float(np.nan)
        def kurtosis(self, x):
            return float(np.nan)
        def shapiro(self, x):
            return (0.0, 1.0)
        def linregress(self, x, y):
            # slope, intercept, r_value, p_value, std_err
            return (0.0, 0.0, 0.0, 1.0, 0.0)
    stats = _StatsFallback()  # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Research-focused styling
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stMetric {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.25rem;
    border-left: 3px solid #0066cc;
}
.metric-container {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
h1, h2, h3 {
    color: #2c3e50;
    font-family: 'Georgia', serif;
}
.research-section {
    border: 1px solid #e1e8ed;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background-color: #fafbfc;
}
</style>
""", unsafe_allow_html=True)

# Constants & dynamic configuration
DEFAULT_API_BASE = "http://localhost:8000"
API_BASE_URL = st.session_state.get('api_base', DEFAULT_API_BASE)
DB_PATH = "data/memory.db"

with st.sidebar:
    st.subheader("⚙️ Configuration")
    new_base = st.text_input("API Base URL", API_BASE_URL, help="Point to a running FastAPI instance.")
    if new_base != API_BASE_URL:
        st.session_state['api_base'] = new_base.strip().rstrip('/')
        st.experimental_rerun()
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=st.session_state.get('auto_refresh', False))
    st.session_state['auto_refresh'] = auto_refresh
    st.divider()
    st.caption("Repro / deterministic runs should set REPRO_MODE=1 on server.")

def safe_api_get(path: str, params=None, timeout: float = 4.0):
    """Best-effort GET against API. Returns (data, error)."""
    url = f"{st.session_state.get('api_base', DEFAULT_API_BASE)}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:120]}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def safe_api_post(path: str, json_body=None, timeout: float = 8.0):
    url = f"{st.session_state.get('api_base', DEFAULT_API_BASE)}{path}"
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def render_health_bar(ok: bool, label: str, detail: str = ""):
    color = "#2ecc71" if ok else "#e74c3c"
    st.markdown(f"<div style='padding:0.4rem;border-radius:4px;background:{color};color:white;font-weight:600;'> {label} {'✅' if ok else '⚠️'} </div>", unsafe_allow_html=True)
    if detail and not ok:
        st.caption(detail)

@st.cache_data(ttl=10)
def get_core_stats():
    data, err = safe_api_get("/stats")
    return data, err

@st.cache_data(ttl=10)
def get_self_learning_status():
    return safe_api_get("/self_learning/status")

@st.cache_data(ttl=10)
def get_training_trigger_preview():
    return safe_api_post("/trigger_training")

def compute_recent_diversity(window_hours: int = 24):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT content, entry_type FROM memory
            WHERE timestamp > datetime('now', ?)
              AND entry_type in ('output','self_output','output_variant')
            """, conn, params=(f'-{window_hours} hours',)
        )
        conn.close()
        if df.empty:
            return {"unique_ratio": 0.0, "count": 0}
        unique_ratio = df['content'].nunique() / max(1, len(df))
        return {"unique_ratio": unique_ratio, "count": len(df)}
    except Exception as e:
        return {"unique_ratio": 0.0, "count": 0, "error": str(e)}

st.markdown("## 🌐 System Overview")
colA, colB, colC, colD = st.columns(4)
core_stats, core_err = get_core_stats()
sl_status, sl_err = get_self_learning_status()
train_preview, train_err = get_training_trigger_preview()
diversity = compute_recent_diversity()

with colA:
    render_health_bar(core_stats is not None, "API /stats", core_err or "")
    st.metric("Total Entries", core_stats.get('total_entries') if core_stats else 0)
with colB:
    render_health_bar(sl_status and sl_status.get('available'), "Self-Learning", (sl_err or sl_status.get('reason') if sl_status else ""))
    if core_stats and 'average_score' in core_stats:
        st.metric("Avg Score", core_stats['average_score'], delta=core_stats.get('recent_average_score'))
with colC:
    if diversity.get('count'):
        st.metric("Output Diversity", f"{diversity['unique_ratio']*100:.1f}%", help="Unique outputs / total last 24h")
    if core_stats and core_stats.get('metrics'):
        r = core_stats['metrics']['retrieval']
        st.metric("Retrieval Hit%", f"{r['hit_rate']*100:.1f}%")
with colD:
    if train_preview:
        readiness = train_preview.get('ready_for_training')
        render_health_bar(readiness, "Training Ready", detail=f"Pairs: {train_preview.get('training_pairs')}" )
    if core_stats and core_stats.get('metrics'):
        pref = core_stats['metrics']['preference']
        st.metric("Pref Pairs", pref.get('pairs_created', 0))

st.markdown("### 🧪 Action Center")
ac1, ac2, ac3 = st.columns([1,1,2])
with ac1:
    st.caption("Start Self-Learning Session")
    iters = st.number_input("Iterations", 1, 20, 5, help="Number of autonomous cycles")
    if st.button("Run Self-Learning", use_container_width=True):
        res, err = safe_api_post("/self_learning/start_session", {"iterations": int(iters)})
        if err:
            st.error(f"Failed: {err}")
        else:
            st.success(f"Session {res['session_id']} avg={res['average_score']}")
with ac2:
    st.caption("Manual Training Cycle Preview")
    if st.button("Preview Training", use_container_width=True):
        st.experimental_rerun()
    if train_preview:
        st.write({k: v for k, v in train_preview.items() if k != 'message'})
with ac3:
    st.caption("Metrics Snapshot")
    if core_stats and core_stats.get('metrics'):
        scoring = core_stats['metrics']['scoring']
        avg_comp = scoring.get('avg_components', {})
        st.json({k: round(v,3) for k,v in avg_comp.items()})

st.divider()

@st.cache_data(ttl=30)
def get_advanced_memory_stats():
    """Get comprehensive statistical analysis from the memory database"""
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
                "temporal_analysis": pd.DataFrame(),
                "score_distribution": pd.DataFrame(),
                "learning_curves": pd.DataFrame(),
                "convergence_metrics": pd.DataFrame(),
                "entropy_analysis": pd.DataFrame(),
                "session_clustering": pd.DataFrame()
            }
        
        # Temporal analysis with statistical features
        temporal_data = pd.read_sql_query("""
            SELECT 
                entry_type,
                datetime(timestamp) as timestamp,
                score,
                session_id,
                length(content) as content_length,
                CASE 
                    WHEN entry_type = 'self_feedback' THEN 1 
                    ELSE 0 
                END as is_self_generated
            FROM memory 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp
        """, conn)
        
        # Score distribution analysis
        score_dist = pd.read_sql_query("""
            SELECT 
                score,
                entry_type,
                datetime(timestamp) as timestamp,
                ROW_NUMBER() OVER (PARTITION BY entry_type ORDER BY timestamp) as sequence_num
            FROM memory 
            WHERE score IS NOT NULL
            ORDER BY timestamp
        """, conn)
        
        # Learning curve data
        learning_curves = pd.read_sql_query("""
            SELECT 
                session_id,
                entry_type,
                score,
                datetime(timestamp) as timestamp,
                ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) as iteration
            FROM memory 
            WHERE session_id IS NOT NULL AND score IS NOT NULL
            ORDER BY session_id, timestamp
        """, conn)
        
        # Convergence analysis
        convergence_data = pd.read_sql_query("""
            SELECT 
                DATE(timestamp) as date,
                entry_type,
                AVG(score) as mean_score,
                COUNT(*) as sample_size,
                MIN(score) as min_score,
                MAX(score) as max_score
            FROM memory 
            WHERE score IS NOT NULL
            GROUP BY DATE(timestamp), entry_type
            ORDER BY date
        """, conn)
        
        # Add standard deviation calculation
        for idx, row in convergence_data.iterrows():
            date_filter = f"DATE(timestamp) = '{row['date']}' AND entry_type = '{row['entry_type']}'"
            std_query = f"""
                SELECT 
                    CASE 
                        WHEN COUNT(*) > 1 THEN 
                            SQRT(SUM((score - {row['mean_score']}) * (score - {row['mean_score']})) / (COUNT(*) - 1))
                        ELSE 0 
                    END as score_std
                FROM memory 
                WHERE {date_filter} AND score IS NOT NULL
            """
            std_result = pd.read_sql_query(std_query, conn)
            convergence_data.at[idx, 'score_std'] = std_result['score_std'].iloc[0] if not std_result.empty else 0
        
        # Content entropy analysis
        entropy_data = pd.read_sql_query("""
            SELECT 
                entry_type,
                length(content) as content_length,
                score,
                datetime(timestamp) as timestamp,
                CASE 
                    WHEN content LIKE '%math%' OR content LIKE '%calculation%' THEN 'mathematical'
                    WHEN content LIKE '%creative%' OR content LIKE '%imagine%' THEN 'creative'
                    WHEN content LIKE '%explain%' OR content LIKE '%define%' THEN 'explanatory'
                    ELSE 'general'
                END as content_category
            FROM memory 
            WHERE timestamp > datetime('now', '-7 days')
        """, conn)
        
        conn.close()
        
        return {
            "temporal_analysis": temporal_data,
            "score_distribution": score_dist,
            "learning_curves": learning_curves,
            "convergence_metrics": convergence_data,
            "entropy_analysis": entropy_data,
            "session_clustering": learning_curves
        }
        
    except Exception as e:
        st.error(f"Database analysis error: {e}")
        return {
            "temporal_analysis": pd.DataFrame(),
            "score_distribution": pd.DataFrame(),
            "learning_curves": pd.DataFrame(),
            "convergence_metrics": pd.DataFrame(),
            "entropy_analysis": pd.DataFrame(),
            "session_clustering": pd.DataFrame()
        }

def calculate_statistical_metrics(data):
    """Calculate advanced statistical metrics for research analysis"""
    metrics = {}
    
    if not data.empty and 'score' in data.columns:
        scores = data['score'].dropna()
        
        if len(scores) > 0:
            # Basic statistics
            metrics['mean'] = scores.mean()
            metrics['std'] = scores.std()
            metrics['variance'] = scores.var()
            metrics['skewness'] = stats.skew(scores)
            metrics['kurtosis'] = stats.kurtosis(scores)
            
            # Distribution analysis
            metrics['q25'] = scores.quantile(0.25)
            metrics['median'] = scores.median()
            metrics['q75'] = scores.quantile(0.75)
            metrics['iqr'] = metrics['q75'] - metrics['q25']
            
            # Normality test
            if len(scores) >= 8:
                try:
                    stat, p_value = stats.shapiro(scores)
                    metrics['shapiro_stat'] = stat
                    metrics['normality_p'] = p_value
                    metrics['is_normal'] = p_value > 0.05
                except:
                    metrics['is_normal'] = False
            
            # Trend analysis
            if len(scores) >= 3:
                x = np.arange(len(scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
                metrics['trend_slope'] = slope
                metrics['trend_r2'] = r_value**2
                metrics['trend_p_value'] = p_value
                metrics['trend_direction'] = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
    
    return metrics

def create_learning_convergence_plot(convergence_data):
    """Create advanced convergence analysis plot"""
    if convergence_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Mean Score Evolution', 'Score Variance Over Time', 
                       'Sample Size Distribution', 'Convergence Rate'],
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Mean score with confidence intervals
    for entry_type in convergence_data['entry_type'].unique():
        type_data = convergence_data[convergence_data['entry_type'] == entry_type]
        
        # Calculate confidence intervals
        ci_upper = type_data['mean_score'] + 1.96 * (type_data['score_std'] / np.sqrt(type_data['sample_size']))
        ci_lower = type_data['mean_score'] - 1.96 * (type_data['score_std'] / np.sqrt(type_data['sample_size']))
        
        fig.add_trace(
            go.Scatter(x=type_data['date'], y=type_data['mean_score'],
                      mode='lines+markers', name=f'{entry_type} (mean)',
                      line=dict(width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=type_data['date'], y=ci_upper,
                      mode='lines', line=dict(width=0),
                      showlegend=False, hoverinfo='skip'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=type_data['date'], y=ci_lower,
                      mode='lines', line=dict(width=0),
                      fill='tonexty', fillcolor=f'rgba(0,100,80,0.2)',
                      showlegend=False, hoverinfo='skip'),
            row=1, col=1
        )
    
    # Plot 2: Variance analysis
    for entry_type in convergence_data['entry_type'].unique():
        type_data = convergence_data[convergence_data['entry_type'] == entry_type]
        fig.add_trace(
            go.Scatter(x=type_data['date'], y=type_data['score_std'],
                      mode='lines+markers', name=f'{entry_type} (std)',
                      line=dict(dash='dash')),
            row=1, col=2
        )
    
    # Plot 3: Sample size
    sample_sizes = convergence_data.groupby('date')['sample_size'].sum()
    fig.add_trace(
        go.Bar(x=sample_sizes.index, y=sample_sizes.values,
               name='Daily Samples', showlegend=False),
        row=2, col=1
    )
    
    # Plot 4: Convergence rate (rolling window analysis)
    if len(convergence_data) > 5:
        convergence_data['rolling_std'] = convergence_data.groupby('entry_type')['mean_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        for entry_type in convergence_data['entry_type'].unique():
            type_data = convergence_data[convergence_data['entry_type'] == entry_type]
            fig.add_trace(
                go.Scatter(x=type_data['date'], y=type_data['rolling_std'],
                          mode='lines', name=f'{entry_type} (convergence)',
                          line=dict(width=3)),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True, 
                      title_text="Learning System Convergence Analysis")
    fig.update_annotations(font_size=12)
    
    return fig

def create_score_distribution_analysis(score_data):
    """Create comprehensive score distribution analysis"""
    if score_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Score Distribution by Type', 'Q-Q Plot (Normality)', 
                       'Temporal Score Evolution', 'Score Autocorrelation'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Distribution comparison
    for entry_type in score_data['entry_type'].unique():
        type_scores = score_data[score_data['entry_type'] == entry_type]['score'].dropna()
        if len(type_scores) > 0:
            fig.add_trace(
                go.Histogram(x=type_scores, name=entry_type, opacity=0.7,
                            nbinsx=20, histnorm='probability density'),
                row=1, col=1
            )
    
    # Plot 2: Q-Q plot for normality assessment
    all_scores = score_data['score'].dropna()
    if len(all_scores) > 3:
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(all_scores)))
        sample_quantiles = np.sort(all_scores)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', name='Observed vs Normal',
                      showlegend=False),
            row=1, col=2
        )
        
        # Add reference line
        min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Normal',
                      line=dict(dash='dash', color='red'), showlegend=False),
            row=1, col=2
        )
    
    # Plot 3: Temporal evolution with rolling statistics
    score_data['timestamp'] = pd.to_datetime(score_data['timestamp'])
    score_data = score_data.sort_values('timestamp')
    
    if len(score_data) > 1:
        score_data['rolling_mean'] = score_data['score'].rolling(window=10, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(x=score_data['timestamp'], y=score_data['score'],
                      mode='markers', name='Raw Scores',
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=score_data['timestamp'], y=score_data['rolling_mean'],
                      mode='lines', name='Rolling Mean',
                      line=dict(width=3, color='red'), showlegend=False),
            row=2, col=1
        )
    
    # Plot 4: Autocorrelation analysis
    if len(all_scores) > 10:
        autocorr = []
        for lag in range(1, min(20, len(all_scores)//2)):
            try:
                # Calculate correlation manually to avoid pandas issues
                shifted = all_scores.shift(lag).dropna()
                original = all_scores[:len(shifted)]
                if len(shifted) > 1 and len(original) > 1:
                    corr = np.corrcoef(original, shifted)[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr.append(0)
            except:
                autocorr.append(0)
        
        lags = list(range(1, len(autocorr) + 1))
        
        fig.add_trace(
            go.Bar(x=lags, y=autocorr, name='Autocorrelation',
                   showlegend=False),
            row=2, col=2
        )
        
        # Add significance threshold
        significance_threshold = 1.96 / np.sqrt(len(all_scores))
        fig.add_hline(y=significance_threshold, line_dash="dash", line_color="red",
                     row=2, col=2)
        fig.add_hline(y=-significance_threshold, line_dash="dash", line_color="red",
                     row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True,
                      title_text="Statistical Distribution Analysis")
    
    return fig

@st.cache_data(ttl=10)
def get_system_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def send_prompt_to_llm(prompt):
    """Send a prompt to the LLM and display the response"""
    try:
        with st.spinner("Sending prompt to LLM..."):
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"query": prompt},
                timeout=30
            )
            
        if response.status_code == 200:
            result = response.json()
            
            # Display the interaction
            st.success("✅ Prompt sent successfully!")
            
            with st.expander("📤 Your Prompt", expanded=True):
                st.write(prompt)
            
            with st.expander("🤖 LLM Response", expanded=True):
                st.write(result.get("response", "No response received"))
            
            # Show any metadata
            if "metadata" in result:
                with st.expander("📊 Response Metadata"):
                    st.json(result["metadata"])
                    
        else:
            st.error(f"❌ Failed to send prompt: {response.status_code}")
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The LLM might be processing a complex query.")
    except Exception as e:
        st.error(f"❌ Error sending prompt: {str(e)}")

def initiate_self_learning(iterations, topic=None):
    """Start a self-learning session"""
    try:
        payload = {
            "iterations": iterations,
            "topic": topic if topic and topic.strip() else None
        }
        with st.spinner(f"Initiating self-learning session ({iterations} iterations)..."):
            result, err = _post_json("/self_learn", payload, timeout=300)
        if err:
            st.error(f"❌ Failed to start self-learning: {err}")
            return
            st.success(f"🚀 Self-learning session started!")
            
            with st.expander("📋 Session Details", expanded=True):
                st.write(f"**Session ID:** {result.get('session_id', 'Unknown')}")
                st.write(f"**Iterations Requested:** {iterations}")
                if topic:
                    st.write(f"**Topic:** {topic}")
                st.write(f"**Status:** {result.get('status', 'Started')}")
                st.write(f"**Estimated Duration:** {result.get('estimated_duration_minutes', 'Unknown')} minutes")
                
            if result.get('status') == 'started':
                st.info(f"📈 Learning session is running in the background. Session ID: {result.get('session_id')}")
                st.info("💡 You can check the system statistics to monitor progress.")
            elif "progress" in result:
                st.info(f"📈 Progress: {result['progress']}")
            
    except Exception as e:
        st.error(f"❌ Error starting self-learning: {str(e)}")

def initiate_training_session():
    """Start a LoRA training session (basic pathway)"""
    try:
        with st.spinner("Starting LoRA training session..."):
            result, err = _post_json("/train", {"mode": "lora"}, timeout=60)
        if err:
            st.error(f"❌ Failed to start LoRA training: {err}")
            return
            
            if result.get('status') == 'insufficient_data':
                st.warning("⚠️ **Insufficient Training Data**")
                st.write(f"**Message:** {result.get('message', 'Need more feedback data')}")
                st.write(f"**Available Training Pairs:** {result.get('training_data_available', 0)}")
                st.write(f"**Good Feedback Entries:** {result.get('good_feedback_entries', 0)}")
                st.info("💡 " + result.get('recommendation', 'Collect more high-quality feedback before training'))
                
                # Option to force training
                if st.button("🚀 Force Training (Override Data Check)"):
                    force_training()
            else:
                st.success("🎯 LoRA Training Initiated!")
                
                with st.expander("🏋️ Training Details", expanded=True):
                    st.write(f"**Session ID:** {result.get('session_id', 'Unknown')}")
                    st.write(f"**Mode:** {result.get('mode', 'LoRA Training')}")
                    st.write(f"**Status:** {result.get('status', 'Started')}")
                    
                    if result.get('mode') == 'lora_async':
                        st.write(f"**Running:** Background (Async)")
                        st.write(f"**Estimated Duration:** {result.get('estimated_duration_minutes', 'Unknown')} minutes")
                        st.info("💡 Training is running in the background. Check training status to monitor progress.")
                    elif result.get('mode') == 'lora_sync':
                        st.write(f"**Running:** Foreground (Sync)")
                        if 'training_result' in result:
                            training_result = result['training_result']
                            if training_result.get('status') == 'completed':
                                st.success("✅ Training completed successfully!")
                                st.write(f"**Training Examples:** {training_result.get('training_examples', 0)}")
                                st.write(f"**Adapter Path:** {training_result.get('adapter_path', 'Unknown')}")
                            else:
                                st.error(f"❌ Training failed: {training_result.get('error', 'Unknown error')}")
                        
        else:
            pass
                
    except Exception as e:
        st.error(f"❌ Error starting LoRA training: {str(e)}")

def force_training():
    """Force LoRA training even with insufficient data (override safety)"""
    try:
        with st.spinner("Force starting LoRA training..."):
            result, err = _post_json("/train", {"mode": "lora", "force_retrain": True}, timeout=120)
        if err:
            st.error(f"❌ Force training failed: {err}")
            return
            st.success("🚀 Forced LoRA Training Started!")
            
            with st.expander("🏋️ Force Training Details", expanded=True):
                st.json(result)
            
    except Exception as e:
        st.error(f"❌ Error with force training: {str(e)}")

def _post_json(path: str, payload: dict, timeout: int = 60):
    """Internal helper to POST JSON and return (result, error)."""
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        if r.status_code != 200:
            try:
                detail = r.json().get('detail', r.text[:120])
            except Exception:
                detail = r.text[:120]
            return None, f"HTTP {r.status_code}: {detail}"
        return r.json(), None
    except Exception as ex:
        return None, str(ex)

def get_recent_responses(limit=10):
    """Get recent LLM responses for feedback"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get recent outputs from the database
        query = """
            SELECT id, content, timestamp, session_id, score
            FROM memory 
            WHERE entry_type IN ('output', 'self_output') 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        if df.empty:
            return []
        
        # Convert to list of dictionaries
        responses = []
        for _, row in df.iterrows():
            responses.append({
                'id': row['id'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'session_id': row['session_id'],
                'current_score': row['score']
            })
        
        return responses
        
    except Exception as e:
        st.error(f"Error retrieving recent responses: {str(e)}")
        return []

def submit_feedback(response_data, score, feedback_text):
    """Submit feedback for a specific response"""
    try:
        payload = {
            "response_id": response_data['id'],
            "score": score,
            "feedback": feedback_text.strip() if feedback_text else None,
            "session_id": response_data.get('session_id')
        }
        
        with st.spinner("Submitting feedback..."):
            response = requests.post(
                f"{API_BASE_URL}/feedback",
                json=payload,
                timeout=10
            )
        
        if response.status_code == 200:
            st.success("💾 Feedback submitted successfully!")
            
            # Also update the database directly
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Insert feedback entry
                cursor.execute("""
                    INSERT INTO memory (entry_type, content, score, session_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    'feedback',
                    feedback_text if feedback_text else f"Score: {score}",
                    score,
                    response_data.get('session_id'),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                # Clear cache to refresh data
                st.cache_data.clear()
                
            except Exception as db_error:
                st.warning(f"Feedback sent to API but database update failed: {str(db_error)}")
                
        else:
            st.error(f"❌ Failed to submit feedback: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error submitting feedback: {str(e)}")

def show_training_status():
    """Display current training status and readiness"""
    try:
        with st.spinner("Checking training status..."):
            response = requests.get(f"{API_BASE_URL}/train/status", timeout=10)
        
        if response.status_code == 200:
            status = response.json()
            
            st.success("📊 Training Status Retrieved!")
            
            # Main status indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Feedback", 
                    status.get('total_feedback_entries', 0),
                    help="All feedback entries in the system"
                )
            
            with col2:
                st.metric(
                    "Good Feedback", 
                    status.get('good_feedback_entries', 0),
                    help="Feedback entries with score ≥ 3.0"
                )
            
            with col3:
                st.metric(
                    "Training Pairs", 
                    status.get('available_training_pairs', 0),
                    help="Available prompt-response pairs for training"
                )
            
            # Readiness status
            if status.get('ready_for_training', False):
                st.success("✅ **Ready for LoRA Training!** Sufficient feedback data available.")
            else:
                st.warning("⚠️ **More data needed** - Collect at least 10 good feedback entries before training.")
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.json(status)
                
        else:
            st.error(f"❌ Failed to get training status: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error checking training status: {str(e)}")

def show_training_history():
    """Display training session history"""
    try:
        with st.spinner("Loading training history..."):
            response = requests.get(f"{API_BASE_URL}/train/history", timeout=10)
        
        if response.status_code == 200:
            history_data = response.json()
            history = history_data.get('training_history', [])
            
            if history:
                st.success(f"📈 Training History ({len(history)} entries)")
                
                # Display history as a table
                history_df = pd.DataFrame(history)
                
                # Format timestamp
                if 'timestamp' in history_df.columns:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show detailed view for recent sessions
                with st.expander("🔍 Detailed Session Info"):
                    for i, session in enumerate(history[:5]):  # Show last 5 sessions
                        st.write(f"**Session {i+1}:** {session.get('session_id', 'Unknown')}")
                        st.write(f"- **Time:** {session.get('timestamp', 'Unknown')}")
                        st.write(f"- **Type:** {session.get('type', 'Unknown')}")
                        st.write(f"- **Status:** {session.get('content', 'No details')}")
                        st.write(f"- **Score:** {session.get('score', 'N/A')}")
                        st.divider()
            else:
                st.info("📭 No training history available yet.")
                
        else:
            st.error(f"❌ Failed to get training history: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error loading training history: {str(e)}")

def validate_adapter():
    """Validate current adapter for model collapse and knowledge retention"""
    with st.spinner("Running adapter validation tests..."):
        try:
            response = requests.get(f"{API_BASE_URL}/train/validate", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "completed":
                    st.success("✅ Adapter validation completed!")
                    
                    health_summary = result.get("health_summary", {})
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        overall_health = health_summary.get("overall_health", "unknown")
                        if overall_health == "healthy":
                            st.success(f"🟢 **Health:** {overall_health.title()}")
                        elif overall_health == "acceptable":
                            st.warning(f"🟡 **Health:** {overall_health.title()}")
                        else:
                            st.error(f"🔴 **Health:** {overall_health.title()}")
                    
                    with col2:
                        knowledge_retained = health_summary.get("knowledge_retained", False)
                        if knowledge_retained:
                            st.success("🧠 **Knowledge:** ✅ Retained")
                        else:
                            st.error("🧠 **Knowledge:** ❌ Lost")
                    
                    with col3:
                        diversity_score = health_summary.get("diversity_score", 0) * 100
                        if diversity_score > 60:
                            st.success(f"🎨 **Diversity:** {diversity_score:.1f}%")
                        elif diversity_score > 40:
                            st.warning(f"🎨 **Diversity:** {diversity_score:.1f}%")
                        else:
                            st.error(f"🎨 **Diversity:** {diversity_score:.1f}%")
                    
                    # Show detailed validation results
                    with st.expander("📊 Detailed Validation Results"):
                        validation_result = result.get("validation_result", {})
                        
                        # Knowledge retention details
                        knowledge_tests = validation_result.get("knowledge_retention", {})
                        if knowledge_tests:
                            st.write("**Knowledge Retention Tests:**")
                            gk_tests = knowledge_tests.get("general_knowledge", [])
                            passed = sum(1 for test in gk_tests if test.get("passed", False))
                            st.write(f"General Knowledge: {passed}/{len(gk_tests)} passed")
                            
                            reasoning_tests = knowledge_tests.get("reasoning", [])
                            reasoning_passed = sum(1 for test in reasoning_tests if test.get("passed", False))
                            st.write(f"Reasoning: {reasoning_passed}/{len(reasoning_tests)} passed")
                        
                        # Diversity metrics
                        diversity_metrics = validation_result.get("diversity_metrics", {})
                        if diversity_metrics:
                            st.write("**Diversity Metrics:**")
                            for metric, value in diversity_metrics.items():
                                if isinstance(value, float):
                                    st.write(f"{metric.replace('_', ' ').title()}: {value:.3f}")
                    
                    # Show recommendations
                    recommendations = result.get("actionable_recommendations", [])
                    if recommendations:
                        st.warning("💡 **Recommendations:**")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    
                    # Safety assessment
                    safe_for_use = health_summary.get("safe_for_continued_use", False)
                    if safe_for_use:
                        st.success("✅ **Assessment:** Safe for continued use")
                    else:
                        st.error("⚠️ **Assessment:** Use with caution or consider retraining")
                
                else:
                    st.error(f"❌ Validation failed: {result.get('message', 'Unknown error')}")
            else:
                st.error(f"❌ Validation request failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Validation timed out - this may indicate model loading issues")
        except Exception as e:
            st.error(f"❌ Validation error: {str(e)}")

def start_advanced_training(min_score: float, max_samples: int, force_retrain: bool):
    """Start advanced LoRA training with custom parameters and safeguards"""
    try:
        payload = {
            "mode": "lora",
            "min_feedback_score": min_score,
            "max_samples": max_samples,
            "force_retrain": force_retrain
        }
        
        with st.spinner("Initiating advanced LoRA training... This may take several minutes."):
            result, err = _post_json("/train", payload, timeout=600)
        if err:
            st.error(f"❌ Advanced LoRA training failed: {err}")
            return
            
            if result.get('status') == 'insufficient_data' and not force_retrain:
                st.warning("⚠️ **Insufficient Training Data**")
                st.write(f"**Message:** {result.get('message', 'Need more feedback')}")
                st.info("💡 Enable 'Force Training' to override this check")
            else:
                st.success("🎉 Advanced LoRA Training Initiated!")
                
                # Display training results
                with st.expander("📋 Training Results", expanded=True):
                    if result.get('status') == 'completed':
                        st.success(f"✅ **Training Status:** {result.get('status', 'Unknown')}")
                        
                        training_result = result.get('training_result', {})
                        if training_result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Training Examples", training_result.get('training_examples', 0))
                            with col2:
                                st.metric("Training Loss", f"{training_result.get('training_loss', 0):.4f}")
                            with col3:
                                st.metric("Duration (min)", f"{training_result.get('training_time_minutes', 0):.1f}")
                            
                            st.info(f"🎯 **Adapter saved to:** {training_result.get('adapter_path', 'Unknown')}")
                            
                    elif result.get('status') == 'started':
                        st.info(f"� **Training Status:** Started in background")
                        st.write(f"**Session ID:** {result.get('session_id')}")
                        st.write(f"**Mode:** {result.get('mode', 'LoRA')}")
                        st.write(f"**Estimated Duration:** {result.get('estimated_duration_minutes', 'Unknown')} minutes")
                        st.info("💡 Training is running in the background. Check training status to monitor progress.")
                    else:
                        st.error(f"❌ Training failed: {result.get('message', 'Unknown error')}")
                
    except requests.exceptions.Timeout:
        st.error("⏱️ Training request timed out. The training may still be running in the background.")
        st.info("💡 Check the training history to see if the session completed.")
    except Exception as e:
        st.error(f"❌ Error starting advanced LoRA training: {str(e)}")

def main():
    # Header
    st.title("AI Learning System Research Analytics")
    st.markdown("**Advanced Performance Metrics and Statistical Analysis**")
    st.markdown("*Research Dashboard for Self-Improving Language Model Evaluation*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # System status
        system_online = get_system_status()
        status_indicator = "🟢" if system_online else "🔴"
        st.markdown(f"**System Status:** {status_indicator} {'Online' if system_online else 'Offline'}")
        
        if not system_online:
            st.error("Cannot proceed with analysis - system offline")
            st.stop()
        
        # Analysis parameters
        st.subheader("Temporal Parameters")
        analysis_window = st.selectbox(
            "Analysis Window",
            ["24 hours", "3 days", "7 days", "30 days"],
            index=2
        )
        
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        st.subheader("Statistical Options")
        show_confidence_intervals = st.checkbox("Show Confidence Intervals", True)
        show_outliers = st.checkbox("Highlight Outliers", True)
        normalize_scores = st.checkbox("Normalize Score Distributions", False)
        
        # Refresh controls
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Interactive LLM Controls
        st.divider()
        st.subheader("🧠 LLM Interaction")
        
        # Prompt input
        prompt_text = st.text_area(
            "Send Prompt to LLM",
            placeholder="Enter your prompt here...",
            height=100,
            help="Send a direct prompt to the language model"
        )
        
        if st.button("📤 Send Prompt", type="primary"):
            if prompt_text.strip():
                send_prompt_to_llm(prompt_text)
            else:
                st.warning("Please enter a prompt")
        
        # Self-learning controls
        st.divider()
        st.subheader("🔄 Self-Learning")
        
        learning_iterations = st.slider("Learning Iterations", 1, 20, 5)
        learning_topic = st.text_input(
            "Learning Topic (optional)",
            placeholder="e.g., mathematics, reasoning, coding"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Self-Learning"):
                initiate_self_learning(learning_iterations, learning_topic)
        
        with col2:
            if st.button("🎯 Start Training"):
                initiate_training_session()
        
        # Advanced LoRA Training section with Safeguards
        st.divider()
        st.subheader("🛡️ LoRA Training with Safeguards")
        
        # Display training health status
        try:
            health_response = requests.get(f"{API_BASE_URL}/train/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                system_health = health_data.get("system_health", "unknown")
                
                if system_health == "healthy":
                    st.success("🟢 Training System: Healthy")
                elif system_health == "concerning":
                    st.warning("🟡 Training System: Some Concerns")
                elif system_health == "unhealthy":
                    st.error("🔴 Training System: Issues Detected")
                else:
                    st.info("⚪ Training System: Status Unknown")
                
                # Show issues and recommendations
                issues = health_data.get("issues", [])
                recommendations = health_data.get("recommendations", [])
                
                if issues or recommendations:
                    with st.expander("⚠️ Health Details"):
                        if issues:
                            st.write("**Issues:**")
                            for issue in issues:
                                st.write(f"• {issue}")
                        if recommendations:
                            st.write("**Recommendations:**")
                            for rec in recommendations:
                                st.write(f"• {rec}")
        except:
            st.warning("Could not fetch training health status")
        
        # Training status with enhanced metrics
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Check Training Status", use_container_width=True):
                show_training_status()
        
        with col2:
            if st.button("� Validate Current Adapter", use_container_width=True):
                validate_adapter()
        
        # Enhanced training controls with safeguards
        with st.expander("🔬 Advanced LoRA Training Options (with Safeguards)", expanded=False):
            st.info("🛡️ **Safeguards Active:** This training includes knowledge retention testing, "
                   "output diversity monitoring, conservative learning rates, and automatic adapter versioning.")
            
            col1, col2 = st.columns(2)
            with col1:
                min_score = st.slider("Minimum Feedback Score", 1.0, 5.0, 3.0, 0.5)
                max_samples = st.number_input("Max Training Samples", 50, 1000, 500, 50)
            
            with col2:
                force_retrain = st.checkbox("Force Training (override safety checks)")
                st.warning("⚠️ Force training bypasses data quality and safety checks")
                
            if st.button("🚀 Start Safeguarded LoRA Training", use_container_width=True, type="primary"):
                start_advanced_training(min_score, max_samples, force_retrain)
        
        # Feedback section
        st.divider()
        st.subheader("💬 Provide Feedback")
        
        # Get recent responses for feedback
        recent_responses = get_recent_responses()
        if recent_responses:
            response_to_feedback = st.selectbox(
                "Select Response to Rate",
                options=range(len(recent_responses)),
                format_func=lambda x: f"Response {x+1}: {recent_responses[x]['content'][:50]}..."
            )
            
            feedback_score = st.slider("Feedback Score", 0.0, 5.0, 2.5, 0.1)
            feedback_text = st.text_area(
                "Feedback Comments",
                placeholder="Optional: Provide detailed feedback...",
                height=80
            )
            
            if st.button("💾 Submit Feedback"):
                submit_feedback(recent_responses[response_to_feedback], feedback_score, feedback_text)
        else:
            st.info("No recent responses available for feedback")
    
    # Load and process data
    data = get_advanced_memory_stats()
    
    if data["temporal_analysis"].empty:
        st.warning("Insufficient data for analysis. System requires more interaction data.")
        
        # Show basic system info
        st.subheader("System Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Database Status", "Connected")
            st.metric("Analysis Window", analysis_window)
        
        with col2:
            st.metric("Confidence Level", f"{confidence_level:.2%}")
            st.metric("Available Tables", "1 (memory)")
        
        st.info("Generate some interactions to begin statistical analysis.")
        return
    
    # Calculate comprehensive metrics
    metrics = calculate_statistical_metrics(data["score_distribution"])
    
    # Main Analysis Dashboard
    st.header("Statistical Overview")
    
    # Key metrics in professional layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Sample Size (n)",
            f"{len(data['score_distribution'])}"
        )
    
    with col2:
        st.metric(
            "Mean Performance",
            f"{metrics.get('mean', 0):.4f}",
            delta=f"σ = {metrics.get('std', 0):.4f}"
        )
    
    with col3:
        st.metric(
            "Distribution Skew",
            f"{metrics.get('skewness', 0):.3f}",
            delta="Asymmetric" if abs(metrics.get('skewness', 0)) > 0.5 else "Symmetric"
        )
    
    with col4:
        normality = "Normal" if metrics.get('is_normal', False) else "Non-Normal"
        p_val = metrics.get('normality_p', 0)
        st.metric(
            "Distribution Test",
            normality,
            delta=f"p = {p_val:.4f}" if p_val else "N/A"
        )
    
    with col5:
        trend = metrics.get('trend_direction', 'stable').title()
        r2 = metrics.get('trend_r2', 0)
        st.metric(
            "Learning Trend",
            trend,
            delta=f"R² = {r2:.3f}" if r2 else "N/A"
        )
    
    # Advanced visualizations
    st.header("Convergence Analysis")
    
    if not data["convergence_metrics"].empty:
        convergence_fig = create_learning_convergence_plot(data["convergence_metrics"])
        st.plotly_chart(convergence_fig, use_container_width=True)
    else:
        st.info("Convergence analysis requires multi-day data")
    
    st.header("Distribution Analysis")
    
    if not data["score_distribution"].empty:
        distribution_fig = create_score_distribution_analysis(data["score_distribution"])
        st.plotly_chart(distribution_fig, use_container_width=True)
    
    # Detailed statistical breakdown
    st.header("Statistical Analysis Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Descriptive Statistics")
        
        if metrics:
            stats_df = pd.DataFrame([
                ["Mean", f"{metrics.get('mean', 0):.6f}"],
                ["Standard Deviation", f"{metrics.get('std', 0):.6f}"],
                ["Variance", f"{metrics.get('variance', 0):.6f}"],
                ["Skewness", f"{metrics.get('skewness', 0):.6f}"],
                ["Kurtosis", f"{metrics.get('kurtosis', 0):.6f}"],
                ["Q1 (25th percentile)", f"{metrics.get('q25', 0):.6f}"],
                ["Median (50th percentile)", f"{metrics.get('median', 0):.6f}"],
                ["Q3 (75th percentile)", f"{metrics.get('q75', 0):.6f}"],
                ["Interquartile Range", f"{metrics.get('iqr', 0):.6f}"],
            ], columns=["Statistic", "Value"])
            
            st.dataframe(stats_df, hide_index=True)
    
    with col2:
        st.subheader("Hypothesis Testing")
        
        if metrics:
            # Normality test results
            if 'shapiro_stat' in metrics:
                st.markdown("**Shapiro-Wilk Normality Test:**")
                st.write(f"• Test Statistic: {metrics['shapiro_stat']:.6f}")
                st.write(f"• p-value: {metrics['normality_p']:.6f}")
                
                alpha = 1 - confidence_level
                if metrics['normality_p'] > alpha:
                    st.success(f"✓ Null hypothesis retained (α = {alpha})")
                    st.write("Data appears normally distributed")
                else:
                    st.warning(f"✗ Null hypothesis rejected (α = {alpha})")
                    st.write("Data deviates from normal distribution")
            
            # Trend analysis
            if 'trend_slope' in metrics:
                st.markdown("**Linear Trend Analysis:**")
                st.write(f"• Slope: {metrics['trend_slope']:.8f}")
                st.write(f"• R-squared: {metrics['trend_r2']:.6f}")
                st.write(f"• p-value: {metrics['trend_p_value']:.6f}")
                
                if metrics['trend_p_value'] < 0.05:
                    st.success("✓ Significant linear trend detected")
                else:
                    st.info("○ No significant linear trend")
    
    # Learning efficiency analysis
    if not data["learning_curves"].empty:
        st.header("Learning Efficiency Analysis")
        
        # Session-based analysis
        session_analysis = data["learning_curves"].groupby('session_id').agg({
            'score': ['count', 'mean', 'std', 'min', 'max'],
            'iteration': 'max'
        }).round(4)
        
        session_analysis.columns = ['_'.join(col).strip() for col in session_analysis.columns.values]
        session_analysis = session_analysis.reset_index()
        
        # Calculate learning rate per session
        learning_rates = []
        for session_id in data["learning_curves"]['session_id'].unique():
            session_data = data["learning_curves"][data["learning_curves"]['session_id'] == session_id]
            if len(session_data) > 1:
                x = session_data['iteration'].values
                y = session_data['score'].values
                try:
                    slope, _, r_value, p_value, _ = stats.linregress(x, y)
                    learning_rates.append({
                        'session_id': session_id,
                        'learning_rate': slope,
                        'r_squared': r_value**2,
                        'significance': p_value
                    })
                except:
                    pass
        
        if learning_rates:
            learning_df = pd.DataFrame(learning_rates)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Learning Rate Distribution")
                fig = px.histogram(learning_df, x='learning_rate', nbins=20,
                                 title="Distribution of Session Learning Rates")
                fig.add_vline(x=learning_df['learning_rate'].mean(), 
                             line_dash="dash", line_color="red",
                             annotation_text="Mean")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Learning Rate Statistics")
                lr_stats = {
                    "Mean Learning Rate": f"{learning_df['learning_rate'].mean():.6f}",
                    "Std Learning Rate": f"{learning_df['learning_rate'].std():.6f}",
                    "Sessions with Positive Learning": f"{(learning_df['learning_rate'] > 0).sum()}",
                    "Sessions with Significant Learning": f"{(learning_df['significance'] < 0.05).sum()}",
                    "Mean R-squared": f"{learning_df['r_squared'].mean():.4f}"
                }
                
                for key, value in lr_stats.items():
                    st.write(f"**{key}:** {value}")
    
    # Model performance comparison
    if not data["temporal_analysis"].empty:
        st.header("Performance Comparison Analysis")
        
        # Compare different entry types
        type_comparison = data["temporal_analysis"].groupby('entry_type').agg({
            'score': ['count', 'mean', 'std', 'min', 'max'],
            'content_length': ['mean', 'std'],
            'is_self_generated': 'sum'
        }).round(4)
        
        type_comparison.columns = ['_'.join(col).strip() for col in type_comparison.columns.values]
        st.subheader("Performance by Entry Type")
        st.dataframe(type_comparison)
        
        # Self-generated vs human-generated comparison
        if 'is_self_generated' in data["temporal_analysis"].columns:
            self_gen = data["temporal_analysis"][data["temporal_analysis"]['is_self_generated'] == 1]['score'].dropna()
            human_gen = data["temporal_analysis"][data["temporal_analysis"]['is_self_generated'] == 0]['score'].dropna()
            
            if len(self_gen) > 0 and len(human_gen) > 0:
                # Statistical comparison
                try:
                    t_stat, p_value = stats.ttest_ind(self_gen, human_gen)
                    
                    st.subheader("Self-Generated vs Human-Generated Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Self-Generated Mean", f"{self_gen.mean():.4f}")
                        st.metric("Human-Generated Mean", f"{human_gen.mean():.4f}")
                    
                    with col2:
                        st.metric("Difference", f"{self_gen.mean() - human_gen.mean():.4f}")
                        pooled_std = np.sqrt((self_gen.var() + human_gen.var()) / 2)
                        effect_size = (self_gen.mean() - human_gen.mean()) / pooled_std if pooled_std > 0 else 0
                        st.metric("Effect Size (Cohen's d)", f"{effect_size:.4f}")
                    
                    with col3:
                        st.metric("t-statistic", f"{t_stat:.4f}")
                        st.metric("p-value", f"{p_value:.6f}")
                        
                        if p_value < 0.05:
                            st.success("Statistically significant difference")
                        else:
                            st.info("No significant difference detected")
                except:
                    st.warning("Could not perform statistical comparison")

if __name__ == "__main__":
    main()
