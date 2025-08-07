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
from scipy import stats
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

# Constants
API_BASE_URL = "http://localhost:8000"
DB_PATH = "data/memory.db"

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
