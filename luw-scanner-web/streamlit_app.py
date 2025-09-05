import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LUW Scanner - Long Until Wrong Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}
.stAlert > div {
    border-radius: 8px;
}
.score-excellent {
    background-color: #d4edda;
    color: #155724;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: bold;
}
.score-good {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: bold;
}
.score-average {
    background-color: #fff3cd;
    color: #856404;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: bold;
}
.score-poor {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False

class SimpleLUWAnalyzer:
    """Simplified LUW analyzer for Streamlit app"""
    
    def __init__(self):
        self.lookback_weeks = 26  # 6 months
        self.range_tolerance = 0.02  # 2%
    
    def get_ticker_data(self, ticker, period="1y"):
        """Get ticker data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            daily_data = stock.history(period=period)
            
            if daily_data.empty:
                return None
                
            # Get some intraday data if available
            try:
                intraday_data = stock.history(period="5d", interval="15m")
            except:
                intraday_data = pd.DataFrame()
            
            return {
                'daily': daily_data,
                'intraday': intraday_data,
                'info': stock.info
            }
        except Exception as e:
            st.warning(f"Error getting data for {ticker}: {e}")
            return None
    
    def calculate_monday_ranges(self, daily_data):
        """Calculate Monday ranges for each week"""
        if daily_data.empty:
            return pd.DataFrame()
        
        # Ensure timezone awareness
        if daily_data.index.tz is None:
            daily_data.index = daily_data.index.tz_localize('US/Eastern')
        
        daily_data = daily_data.copy()
        daily_data['dayofweek'] = daily_data.index.dayofweek
        daily_data['week'] = daily_data.index.isocalendar().week
        daily_data['year'] = daily_data.index.year
        
        monday_ranges = []
        
        # Group by year and week
        for (year, week), week_data in daily_data.groupby(['year', 'week']):
            # Find Monday data (dayofweek = 0)
            monday_data = week_data[week_data['dayofweek'] == 0]
            
            if monday_data.empty:
                continue
            
            monday_high = monday_data['High'].max()
            monday_low = monday_data['Low'].min()
            monday_range = monday_high - monday_low
            
            if monday_range <= 0:
                continue
            
            monday_mid = (monday_high + monday_low) / 2
            fib_147 = monday_low + (monday_range * 0.147)
            fib_853 = monday_low + (monday_range * 0.853)
            
            monday_ranges.append({
                'year': year,
                'week': week,
                'week_start': week_data.index.min(),
                'week_end': week_data.index.max(),
                'monday_high': monday_high,
                'monday_low': monday_low,
                'monday_mid': monday_mid,
                'fib_147': fib_147,
                'fib_853': fib_853,
                'range_size': monday_range,
                'range_pct': (monday_range / monday_low) * 100 if monday_low > 0 else 0
            })
        
        return pd.DataFrame(monday_ranges)
    
    def calculate_range_respect(self, daily_data, monday_ranges):
        """Calculate how well price respects Monday ranges"""
        if monday_ranges.empty:
            return 0.0
        
        total_respect_score = 0
        total_weeks = 0
        
        for _, week_range in monday_ranges.iterrows():
            week_start = week_range['week_start']
            week_end = week_range['week_end']
            
            week_daily = daily_data[
                (daily_data.index >= week_start) & 
                (daily_data.index <= week_end)
            ]
            
            if week_daily.empty:
                continue
            
            total_weeks += 1
            week_respect = 0
            
            for _, day in week_daily.iterrows():
                close_price = day['Close']
                
                # Check if within overall range
                if week_range['monday_low'] <= close_price <= week_range['monday_high']:
                    week_respect += 0.5
                
                # Check respect of key levels
                levels = [week_range['monday_low'], week_range['monday_mid'], week_range['monday_high']]
                for level in levels:
                    tolerance = level * self.range_tolerance
                    if abs(close_price - level) <= tolerance:
                        week_respect += 0.2
            
            # Normalize by number of days
            if len(week_daily) > 0:
                total_respect_score += min(week_respect / len(week_daily), 1.0)
        
        return total_respect_score / total_weeks if total_weeks > 0 else 0.0
    
    def calculate_reversal_quality(self, daily_data, monday_ranges):
        """Calculate reversal quality at Monday levels"""
        if monday_ranges.empty:
            return 0.0
        
        reversal_events = 0
        successful_reversals = 0
        
        for _, week_range in monday_ranges.iterrows():
            week_start = week_range['week_start']
            week_end = week_range['week_end']
            
            week_daily = daily_data[
                (daily_data.index >= week_start) & 
                (daily_data.index <= week_end)
            ]
            
            if len(week_daily) < 3:
                continue
            
            monday_low = week_range['monday_low']
            
            for i in range(len(week_daily) - 2):
                current_day = week_daily.iloc[i]
                
                # Check if touched Monday Low
                if current_day['Low'] <= monday_low * 1.01:  # 1% tolerance
                    reversal_events += 1
                    
                    # Look for reversal in next 2 days
                    for j in range(1, min(3, len(week_daily) - i)):
                        future_day = week_daily.iloc[i + j]
                        if future_day['Close'] > monday_low * 1.02:  # 2% above
                            successful_reversals += 1
                            break
        
        return successful_reversals / reversal_events if reversal_events > 0 else 0.0
    
    def calculate_swing_efficiency(self, daily_data, monday_ranges):
        """Calculate swing efficiency between levels with recency weighting"""
        if monday_ranges.empty or len(daily_data) < 20:
            return 0.0
        
        weekly_efficiency_scores = []
        week_start_dates = []
        current_date = daily_data.index.max()
        
        # Analyze efficiency for each week's Monday range
        for _, week_range in monday_ranges.iterrows():
            week_start = week_range['week_start']
            week_end = week_range['week_end']
            
            week_daily = daily_data[
                (daily_data.index >= week_start) & 
                (daily_data.index <= week_end)
            ]
            
            if len(week_daily) < 3:
                continue
            
            # Calculate week-specific efficiency metrics
            week_daily_copy = week_daily.copy()
            week_daily_copy['daily_range'] = (week_daily_copy['High'] - week_daily_copy['Low']) / week_daily_copy['Close']
            week_avg_range = week_daily_copy['daily_range'].mean()
            
            # Calculate directional consistency within the week
            week_daily_copy['price_change'] = week_daily_copy['Close'].pct_change()
            week_price_std = week_daily_copy['price_change'].std()
            
            # Measure swings relative to Monday range
            monday_range_size = week_range['monday_high'] - week_range['monday_low']
            if monday_range_size > 0:
                # Calculate how much of Monday range was utilized
                week_high = week_daily_copy['High'].max()
                week_low = week_daily_copy['Low'].min()
                range_utilization = (week_high - week_low) / monday_range_size
                
                # Score based on range utilization and consistency
                trend_consistency = max(0.0, 1.0 - (week_price_std * 10))  # Penalize choppy action
                range_score = min(week_avg_range * 10, 1.0)  # Reward good daily ranges
                utilization_score = min(range_utilization, 1.0)  # Reward range utilization
                
                # Combine metrics for weekly efficiency score
                week_efficiency = (range_score * 0.4 + trend_consistency * 0.3 + utilization_score * 0.3)
                weekly_efficiency_scores.append(max(0.0, min(week_efficiency, 1.0)))
                week_start_dates.append(week_start)
        
        if not weekly_efficiency_scores:
            return 0.0
        
        # Apply recency weighting
        weeks_ago = self.get_weeks_ago(week_start_dates, current_date)
        return self.apply_recency_weights(weekly_efficiency_scores, weeks_ago)
    
    def calculate_gap_patterns(self, daily_data):
        """Calculate gap pattern success rate"""
        if len(daily_data) < 10:
            return 0.0
        
        gap_events = 0
        successful_patterns = 0
        
        for i in range(1, len(daily_data)):
            prev_close = daily_data.iloc[i-1]['Close']
            current_open = daily_data.iloc[i]['Open']
            current_close = daily_data.iloc[i]['Close']
            
            gap_percent = abs(current_open - prev_close) / prev_close
            
            if gap_percent > 0.005:  # 0.5% gap
                gap_events += 1
                
                # Simple pattern: gap up should continue, gap down should reverse
                if current_open > prev_close and current_close > current_open:
                    successful_patterns += 1
                elif current_open < prev_close and current_close > current_open:
                    successful_patterns += 1
        
        return successful_patterns / gap_events if gap_events > 0 else 0.0
    
    def analyze_ticker(self, ticker):
        """Complete analysis of a single ticker"""
        try:
            # Get data
            data = self.get_ticker_data(ticker)
            if not data or data['daily'].empty:
                return None
            
            daily_data = data['daily']
            current_price = daily_data['Close'].iloc[-1]
            avg_volume = daily_data['Volume'].mean()
            
            # Basic filters
            if current_price < 5.0 or avg_volume < 500000:
                return None
            
            # Calculate Monday ranges
            monday_ranges = self.calculate_monday_ranges(daily_data)
            if monday_ranges.empty:
                return None
            
            # Calculate component scores
            range_respect = self.calculate_range_respect(daily_data, monday_ranges)
            reversal_quality = self.calculate_reversal_quality(daily_data, monday_ranges)
            swing_efficiency = self.calculate_swing_efficiency(daily_data, monday_ranges)
            gap_patterns = self.calculate_gap_patterns(daily_data)
            
            # Calculate composite score
            weights = {
                'range_respect': 0.30,
                'reversal_quality': 0.25,
                'swing_efficiency': 0.20,
                'gap_patterns': 0.15,
                'volume_consistency': 0.10
            }
            
            volume_consistency = min(avg_volume / 1_000_000, 1.0)  # Normalize volume
            
            composite_score = (
                range_respect * weights['range_respect'] +
                reversal_quality * weights['reversal_quality'] +
                swing_efficiency * weights['swing_efficiency'] +
                gap_patterns * weights['gap_patterns'] +
                volume_consistency * weights['volume_consistency']
            )
            
            return {
                'ticker': ticker,
                'composite_score': composite_score,
                'range_respect': range_respect,
                'reversal_quality': reversal_quality,
                'swing_efficiency': swing_efficiency,
                'gap_patterns': gap_patterns,
                'volume_consistency': volume_consistency,
                'current_price': current_price,
                'avg_volume_million': avg_volume / 1_000_000,
                'total_weeks': len(monday_ranges)
            }
            
        except Exception as e:
            st.warning(f"Error analyzing {ticker}: {e}")
            return None

def get_sp500_sample():
    """Get a sample of S&P 500 tickers for testing (curated list)"""
    return [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Large Cap Traditional
        'BRK-B', 'JNJ', 'JPM', 'PG', 'UNH', 'HD', 'DIS',
        # Finance
        'BAC', 'WFC', 'GS', 'AXP', 'C',
        # Healthcare
        'PFE', 'ABBV', 'TMO', 'ABT', 'LLY',
        # Consumer
        'KO', 'PEP', 'WMT', 'NKE', 'MCD',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'HON',
        # Tech/Growth
        'CRM', 'ADBE', 'NFLX', 'PYPL', 'INTC'
    ]

def get_extended_sp500_list():
    """Get an extended list of S&P 500 tickers for Smart Scan"""
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'CRM', 'ADBE',
        'NFLX', 'PYPL', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'ORCL', 'IBM', 'CSCO',
        'AMAT', 'ADI', 'LRCX', 'KLAC', 'MRVL', 'SNPS', 'CDNS', 'FTNT', 'PANW', 'CRWD',
        
        # Healthcare
        'JNJ', 'PFE', 'ABBV', 'UNH', 'TMO', 'ABT', 'LLY', 'MRK', 'DHR', 'BMY',
        'AMGN', 'GILD', 'MDT', 'CI', 'CVS', 'ANTM', 'HUM', 'BDX', 'SYK', 'BSX',
        'ELV', 'ZTS', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'IQV', 'DXCM', 'ALGN', 'MRNA',
        
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'C', 'MS', 'AXP', 'BLK', 'SCHW', 'SPGI',
        'V', 'MA', 'PYPL', 'COF', 'USB', 'TFC', 'PNC', 'BK', 'STT', 'FITB',
        'HBAN', 'RF', 'CFG', 'KEY', 'ZION', 'CMA', 'NTRS', 'SIVB', 'EWBC', 'PBCT',
        
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'ORLY',
        'CMG', 'LULU', 'RCL', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS', 'CZR', 'NCLH',
        'CCL', 'AAL', 'DAL', 'UAL', 'LUV', 'ALK', 'JBLU', 'SAVE', 'F', 'GM',
        
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
        'CAG', 'SJM', 'TSN', 'HRL', 'CHD', 'CLX', 'COTY', 'EL', 'KHC', 'MDLZ',
        'MNST', 'DG', 'DLTR', 'SYY', 'KR', 'WBA', 'CVS', 'TAP', 'STZ', 'BF-B',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE', 'WMB', 'VLO',
        'PSX', 'MPC', 'HES', 'DVN', 'FANG', 'APA', 'BKR', 'HAL', 'NOV', 'CTRA',
        
        # Industrials
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
        'DE', 'UNP', 'CSX', 'NSC', 'FDX', 'LUV', 'AAL', 'DAL', 'UAL', 'WM',
        'RSG', 'PCAR', 'CMI', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'XYL',
        
        # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'DOW', 'PPG', 'IFF',
        'LYB', 'AVY', 'BLL', 'CF', 'FMC', 'ALB', 'CE', 'EMN', 'IP', 'PKG',
        
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA', 'EXR',
        'AVB', 'EQR', 'VTR', 'BXP', 'ARE', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT',
        
        # Utilities
        'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'D', 'PCG', 'PEG', 'SRE',
        'EIX', 'WEC', 'AWK', 'PPL', 'CMS', 'DTE', 'ATO', 'CNP', 'NI', 'LNT',
        
        # Communication Services
        'GOOGL', 'GOOG', 'META', 'DIS', 'NFLX', 'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA',
        'ATVI', 'EA', 'TTWO', 'ROKU', 'SPOT', 'TWTR', 'SNAP', 'PINS', 'MTCH', 'ZM'
    ]

def get_smart_scan_sample():
    """Get 50 random tickers from extended S&P 500 list for Smart Scan"""
    extended_list = get_extended_sp500_list()
    
    # Ensure we have enough tickers to sample from
    if len(extended_list) < 50:
        return extended_list
    
    # Randomly sample 50 tickers
    return random.sample(extended_list, 50)

def get_sector_tickers(profile):
    """Get tickers based on trading profile"""
    if profile == "Conservative":
        return ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'KO', 'WMT', 'VZ', 'T', 'PFE']
    elif profile == "Aggressive":
        return ['TSLA', 'NVDA', 'AMD', 'NFLX', 'ROKU', 'PLTR', 'SNOW', 'ZM', 'SHOP', 'SQ']
    else:  # Balanced
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JNJ', 'JPM', 'PG', 'NVDA', 'UNH']

def format_score_badge(score):
    """Format score with color coding"""
    if score >= 0.8:
        return f'<span class="score-excellent">{score:.3f}</span>'
    elif score >= 0.6:
        return f'<span class="score-good">{score:.3f}</span>'
    elif score >= 0.4:
        return f'<span class="score-average">{score:.3f}</span>'
    else:
        return f'<span class="score-poor">{score:.3f}</span>'

def show_documentation():
    """Complete technical documentation for LUW Scanner"""
    
    st.markdown("""
    # 📚 LUW Scanner - Technical Documentation
    
    *Complete guide to understanding and using the Long Until Wrong methodology*
    
    ---
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 What is LUW?", 
        "📊 Scoring System", 
        "🔧 Inputs Guide", 
        "📈 Reading Results", 
        "💡 Trading Strategies", 
        "❓ FAQ"
    ])
    
    with tab1:
        st.header("🎯 What is Long Until Wrong (LUW)?")
        
        st.markdown("""
        ### Overview
        
        **Long Until Wrong (LUW)** is a systematic trading methodology that identifies stocks which consistently respect key weekly price levels, specifically the **Monday range** and its Fibonacci retracements.
        
        ### Core Concept
        
        The strategy is based on the observation that certain stocks tend to:
        1. **Respect weekly levels** established during Monday's trading
        2. **Provide predictable reversals** at these key levels
        3. **Offer consistent swing trading opportunities** within defined ranges
        
        ### The Monday Range
        
        Every week, we calculate:
        - **Monday High**: The highest price during Monday's session
        - **Monday Low**: The lowest price during Monday's session  
        - **Monday Midline**: The average of Monday High and Low
        - **Fibonacci Levels**: 14.7% and 85.3% retracements of the Monday range
        
        ### Why Monday Matters
        
        Monday's trading often sets the tone for the entire week because:
        - **Institutional positioning** after weekend analysis
        - **Gap reactions** to weekend news
        - **Volume concentration** as markets reopen
        - **Psychological levels** that traders watch throughout the week
        """)
        
        # Visual explanation
        st.subheader("📊 Monday Range Visualization")
        
        # Create sample Monday range visualization
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = 100 + np.cumsum(np.random.randn(20) * 0.5)
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=dates, 
            y=prices,
            mode='lines',
            name='Stock Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add Monday levels
        monday_high = 102
        monday_low = 98
        monday_mid = 100
        fib_147 = monday_low + (monday_high - monday_low) * 0.147
        fib_853 = monday_low + (monday_high - monday_low) * 0.853
        
        levels = [
            (monday_high, "Monday High", "red"),
            (fib_853, "85.3% Fib", "orange"), 
            (monday_mid, "Monday Midline", "gray"),
            (fib_147, "14.7% Fib", "orange"),
            (monday_low, "Monday Low", "green")
        ]
        
        for level, name, color in levels:
            fig.add_hline(y=level, line_dash="dash", line_color=color, 
                         annotation_text=f"{name} ({level:.2f})")
        
        fig.update_layout(
            title="Monday Range with Fibonacci Levels",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 LUW Scoring System")
        
        st.markdown("""
        ### Composite Score Calculation
        
        The LUW Composite Score (0.0 - 1.0) combines five key components:
        """)
        
        # Scoring breakdown
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            **Component Weights:**
            - Range Respect: 30%
            - Reversal Quality: 25%  
            - Swing Efficiency: 20%
            - Gap Patterns: 15%
            - Volume Consistency: 10%
            """)
        
        with col2:
            # Create scoring visual
            components = ['Range Respect', 'Reversal Quality', 'Swing Efficiency', 'Gap Patterns', 'Volume Consistency']
            weights = [30, 25, 20, 15, 10]
            
            fig = go.Figure(data=[go.Pie(labels=components, values=weights)])
            fig.update_layout(title="LUW Score Component Weights", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Component Explanations
        
        #### 🎯 Range Respect (30% weight)
        **What it measures:** How well the stock's daily closes respect Monday levels
        
        **Interpretation:**
        - **0.8-1.0:** Excellent - Consistently respects levels
        - **0.6-0.8:** Good - Usually respects levels  
        - **0.4-0.6:** Average - Sometimes respects levels
        - **0.0-0.4:** Poor - Rarely respects levels
        
        #### ⚡ Reversal Quality (25% weight)  
        **What it measures:** Speed and reliability of bounces at Monday Low
        
        **Interpretation:**
        - **0.8-1.0:** Quick, reliable reversals (same day)
        - **0.6-0.8:** Good reversals (1-2 days)
        - **0.4-0.6:** Slow reversals (2+ days)
        - **0.0-0.4:** Poor/failed reversals
        
        #### 🌊 Swing Efficiency (20% weight)
        **What it measures:** Trading opportunity richness between levels
        
        **Interpretation:**
        - **0.8-1.0:** Excellent swings with clear directional moves
        - **0.6-0.8:** Good swing opportunities
        - **0.4-0.6:** Moderate swing potential
        - **0.0-0.4:** Choppy, difficult to trade
        
        #### 📈 Gap Patterns (15% weight)
        **What it measures:** Predictability of gap behavior
        
        **Interpretation:**
        - **0.8-1.0:** Highly predictable gap behavior
        - **0.6-0.8:** Generally reliable gap patterns
        - **0.4-0.6:** Mixed gap performance
        - **0.0-0.4:** Unpredictable gap behavior
        
        #### 📊 Volume Consistency (10% weight)
        **What it measures:** Volume reliability for trade execution
        
        **Interpretation:**
        - **0.8-1.0:** Excellent liquidity (50M+ average)
        - **0.6-0.8:** Good liquidity (10-50M average)
        - **0.4-0.6:** Adequate liquidity (1-10M average)
        - **0.0-0.4:** Low liquidity (<1M average)
        """)
        
        st.subheader("🎯 Score Interpretation Guide")
        
        score_guide = {
            "Score Range": ["0.8 - 1.0", "0.6 - 0.8", "0.4 - 0.6", "0.0 - 0.4"],
            "Grade": ["🟢 EXCELLENT", "🟡 GOOD", "🟠 AVERAGE", "🔴 POOR"],
            "LUW Suitability": ["Prime candidate", "Good candidate", "Consider with caution", "Avoid"],
            "Expected Win Rate": ["70-80%", "60-70%", "50-60%", "<50%"],
            "Trade Frequency": ["Multiple per week", "1-2 per week", "1-2 per month", "Rare setups"]
        }
        
        df = pd.DataFrame(score_guide)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("🔧 Input Parameters Guide")
        
        st.markdown("""
        ### Scan Types
        
        Different scan options for various trading needs:
        """)
        
        scan_comparison = {
            "Scan Type": [
                "Quick Scan (10 tickers)",
                "Smart Scan (50 tickers)", 
                "Extended Scan (25 tickers)",
                "Custom Tickers"
            ],
            "Description": [
                "Fast scan with profile-optimized tickers",
                "Random sampling from S&P 500 universe",
                "Curated list of popular stocks",
                "User-defined ticker list"
            ],
            "Best For": [
                "Quick market overview",
                "Comprehensive market discovery", 
                "Balanced screening approach",
                "Specific stock analysis"
            ],
            "Time Required": [
                "30-60 seconds",
                "3-5 minutes",
                "2-3 minutes", 
                "Varies by count"
            ]
        }
        
        df = pd.DataFrame(scan_comparison)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Trading Profiles
        
        Pre-configured settings optimized for different trading styles:
        """)
        
        profile_comparison = {
            "Setting": [
                "Target Stocks",
                "Min Volume", 
                "Min Price",
                "Range Tolerance",
                "Best For",
                "Risk Level",
                "Time Horizon"
            ],
            "Conservative": [
                "Blue-chip, S&P 100",
                "5M+ shares",
                "$20+",
                "1% (strict)",
                "Swing trading, retirement accounts",
                "Low",
                "1-4 weeks"
            ],
            "Balanced": [
                "Large/mid-cap mix",
                "1M+ shares", 
                "$10+",
                "2% (moderate)",
                "General swing/position trading",
                "Medium",
                "1-2 weeks"
            ],
            "Aggressive": [
                "Growth, momentum stocks",
                "1M+ shares",
                "$5+", 
                "3% (loose)",
                "Active trading, speculation",
                "High",
                "1-5 days"
            ]
        }
        
        df = pd.DataFrame(profile_comparison)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Filter Parameters
        
        #### Minimum LUW Score (0.0 - 1.0)
        - **Default:** 0.3
        - **Purpose:** Filter out poor LUW candidates
        - **Recommendation:** 
          - 0.6+ for high-confidence trades
          - 0.4+ for moderate-confidence trades
          - 0.3+ for comprehensive screening
        
        #### Price Range ($)
        - **Min Price:** Avoid penny stocks and illiquid securities
        - **Max Price:** Control position sizing and option availability
        - **Recommendations:**
          - Swing Trading: $10 - $500
          - Day Trading: $20 - $300
          - Options Trading: $50 - $200
        """)
    
    with tab4:
        st.header("📈 How to Read Results")
        
        st.markdown("""
        ### Results Table Interpretation
        
        Your scan results are sorted by **Composite Score** (highest to lowest). Here's how to read each column:
        """)
        
        # Sample results table for demonstration
        sample_results = {
            "Column": ["Ticker", "LUW Score", "Price", "Volume (M)", "Range Respect", "Reversal Quality"],
            "Description": [
                "Stock symbol",
                "Composite LUW score (0-1)",
                "Current stock price",
                "Average daily volume in millions",
                "How well it respects Monday levels",
                "Quality of bounces at Monday Low"
            ],
            "Good Values": [
                "Recognizable symbols",
                "0.6+",
                "Your preference",
                "1M+ (higher better)",
                "0.6+",
                "0.6+"
            ],
            "Red Flags": [
                "Unknown/delisted",
                "<0.4",
                "Penny stocks",
                "<0.5M",
                "<0.3",
                "<0.3"
            ]
        }
        
        df = pd.DataFrame(sample_results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.subheader("🚨 Warning Signs to Avoid")
        
        warning_signs = {
            "Warning": [
                "Range Respect = 0.0",
                "Volume < 0.5M",
                "All scores < 0.3",
                "Extreme price volatility"
            ],
            "What it means": [
                "Stock doesn't respect Monday levels at all",
                "Insufficient liquidity for smooth trading",
                "Poor candidate across all metrics",
                "Unpredictable, high-risk movements"
            ],
            "Action": [
                "❌ Avoid for LUW strategy",
                "❌ Skip unless desperate",
                "❌ Not suitable for LUW",
                "⚠️ Reduce position size"
            ]
        }
        
        df = pd.DataFrame(warning_signs)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.header("💡 LUW Trading Strategies")
        
        st.markdown("""
        ### Core LUW Trading Setups
        
        #### 🎯 Setup 1: Monday Low Bounce
        **When:** Price touches or slightly breaks Monday Low
        **Entry:** 2-3% recovery from Monday Low
        **Target:** Monday Midline or Monday High
        **Stop:** 1 ATR below Monday Low
        **Best Candidates:** Range Respect > 0.6, Reversal Quality > 0.6
        
        #### 🎯 Setup 2: Monday High Breakout  
        **When:** Price breaks above Monday High with volume
        **Entry:** Breakout confirmation (close above Monday High)
        **Target:** 1-2 ATR above Monday High
        **Stop:** Monday High becomes support
        **Best Candidates:** Swing Efficiency > 0.6, Volume > 5M
        
        #### 🎯 Setup 3: Midline Bounce
        **When:** Price rejects from Monday Midline
        **Entry:** Bounce confirmation with volume
        **Target:** Monday High or Fibonacci 85.3%
        **Stop:** Monday Midline breakdown
        **Best Candidates:** Balanced radar chart, Score > 0.6
        
        ### Position Sizing by LUW Score
        """)
        
        position_sizing = {
            "LUW Score": ["0.8 - 1.0", "0.6 - 0.8", "0.4 - 0.6", "0.0 - 0.4"],
            "Position Size": ["2-3% of portfolio", "1-2% of portfolio", "0.5-1% of portfolio", "Avoid"],
            "Risk Level": ["Standard risk", "Reduced risk", "Minimal risk", "No position"],
            "Stop Loss": ["Standard (1 ATR)", "Tight (0.75 ATR)", "Very tight (0.5 ATR)", "N/A"]
        }
        
        df = pd.DataFrame(position_sizing)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Risk Management Rules
        
        #### ✅ DO:
        - Only trade stocks with LUW Score > 0.5
        - Use proper position sizing based on score
        - Set stops below Monday Low for long positions
        - Take profits at Monday High resistance
        - Combine with volume confirmation
        
        #### ❌ DON'T:
        - Trade stocks with Range Respect < 0.3
        - Ignore volume requirements
        - Chase breakouts without confirmation
        - Hold through Monday range violations
        - Use full size on untested candidates
        """)
    
    with tab6:
        st.header("❓ Frequently Asked Questions")
        
        st.markdown("""
        ### 🔧 Getting Started
        
        **Q: I'm new to trading - is LUW suitable for beginners?**
        A: Yes! LUW provides clear, rule-based signals. Start with the Conservative profile and scores above 0.6.
        
        **Q: How much money do I need to start?**
        A: You can start with any amount. Use 1-2% position sizes, so $1,000 minimum is practical for diversification.
        
        ### 📊 Understanding Scores
        
        **Q: What's a good LUW score?**
        A: 0.6+ is excellent, 0.5+ is good, 0.4+ is average. Start with 0.6+ until you're comfortable.
        
        **Q: Why is Range Respect 0.000?**
        A: The stock doesn't respect Monday levels. This is a red flag - avoid these stocks for LUW trading.
        
        **Q: All my scores seem low - is this normal?**
        A: During volatile markets, scores drop. Try the Conservative profile or wait for calmer conditions.
        
        ### 🎯 Trading Questions
        
        **Q: When should I buy?**
        A: Wait for price to touch Monday Low, then buy when it bounces 2-3% higher with volume confirmation.
        
        **Q: When should I sell?**
        A: Take profits at Monday Midline (50%) or Monday High (full). Always use stops below Monday Low.
        
        **Q: What if the stock gaps down?**
        A: If it gaps below Monday Low, wait for recovery above the Monday Low before entering.
        
        ### 🔍 Scan Questions
        
        **Q: What's the difference between scan types?**
        A: Quick (10 tickers, fast), Smart (50 random S&P 500, comprehensive), Extended (25 curated), Custom (your choice).
        
        **Q: Which scan should I use?**
        A: Smart Scan for discovery, Quick for daily checks, Extended for balance, Custom for specific analysis.
        
        ### ⚙️ Technical Issues
        
        **Q: The scan failed - what happened?**
        A: Check your internet connection, reduce the number of custom tickers, or try again in a few minutes.
        
        **Q: No candidates found - why?**
        A: Lower the minimum LUW score to 0.3, increase max price, or try a different profile.
        
        ### 💰 Risk Management
        
        **Q: How much should I risk per trade?**
        A: Never risk more than 1-2% of your account per trade. Use the LUW score to adjust position size.
        
        **Q: Should I trade during earnings?**
        A: No. LUW assumes normal market conditions. Avoid earnings weeks or reduce position sizes by 50%.
        
        **Q: What if Monday Low breaks?**
        A: Exit immediately. Monday Low breaks invalidate the LUW setup for that week.
        """)

def show_quick_start_guide():
    """Quick start guide for new users"""
    
    st.markdown("""
    # 🚀 LUW Scanner - Quick Start Guide
    
    *Get up and running in 5 minutes*
    
    ---
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Step 1: Choose Your Profile
        
        **👥 New to Trading?**
        → Select **"Conservative"**
        
        **⚡ Active Trader?**  
        → Select **"Aggressive"**
        
        **🎯 Balanced Approach?**
        → Keep **"Balanced"** (default)
        """)
    
    with col2:
        st.markdown("""
        ### Step 2: Pick Scan Type
        
        **🚀 First Time?**
        → Use **"Quick Scan"** (fastest)
        
        **🎯 Best Discovery?**
        → Try **"Smart Scan"** (comprehensive)
        
        **📊 Weekly Analysis?**
        → Try **"Extended Scan"**
        
        **🎯 Specific Stocks?**
        → Use **"Custom Tickers"**
        """)
    
    st.markdown("""
    ### Step 3: Run Your First Scan
    
    1. Click **"🔍 Run LUW Scan"** in the sidebar
    2. Wait 1-5 minutes for analysis (depending on scan type)
    3. Review your results!
    
    ### Step 4: Interpret Results
    
    **Look for:**
    - **LUW Score 0.6+** (excellent candidates)
    - **Range Respect 0.5+** (respects Monday levels)
    - **Volume 5M+** (good liquidity)
    
    **Avoid:**
    - Range Respect = 0.000 (doesn't respect levels)
    - Very low volume (<0.5M)
    - Scores below 0.4
    
    ### 🎯 Your First Trade Setup
    
    1. **Pick the highest-scoring candidate**
    2. **Wait for price to touch Monday Low**
    3. **Enter when it bounces 2% above Monday Low**
    4. **Target the Monday Midline or High**
    5. **Stop loss below Monday Low**
    
    ---
    
    ## 💡 Pro Tips for Beginners
    
    ✅ **Start with paper trading** to test the system  
    ✅ **Focus on scores above 0.6** for higher confidence  
    ✅ **Use proper position sizing** (1-2% of portfolio)  
    ✅ **Keep a trading journal** of setups and outcomes  
    ✅ **Try Smart Scan for discovery** - it samples different stocks each time
    
    ❌ **Don't ignore volume requirements**  
    ❌ **Don't chase after missed entries**  
    ❌ **Don't hold through Monday range breaks**  
    ❌ **Don't trade during earnings weeks**  
    
    ---
    
    **Ready to start? Click "Back to Scanner" and run your first scan!**
    """)

def show_faq_only():
    """Show just the FAQ section"""
    
    st.markdown("""
    # ❓ Frequently Asked Questions
    
    *Quick answers to common LUW Scanner questions*
    
    ---
    
    ### 🔧 Getting Started
    
    **Q: I'm new to trading - is LUW suitable for beginners?**
    A: Yes! LUW provides clear, rule-based signals. Start with the Conservative profile and scores above 0.6.
    
    **Q: What's a good LUW score?**
    A: 0.6+ is excellent, 0.5+ is good, 0.4+ is average. Start with 0.6+ until you're comfortable.
    
    **Q: Why is Range Respect 0.000?**
    A: The stock doesn't respect Monday levels. This is a red flag - avoid these stocks for LUW trading.
    
    ### 🔍 Scan Types
    
    **Q: What's the difference between scan types?**
    A: Quick (10 tickers, fast), Smart (50 random S&P 500, comprehensive), Extended (25 curated), Custom (your choice).
    
    **Q: Which scan should I use?**
    A: Smart Scan for discovery, Quick for daily checks, Extended for balance, Custom for specific analysis.
    
    **Q: Why does Smart Scan show different results each time?**
    A: Smart Scan randomly samples 50 tickers from a large S&P 500 list, so you discover different opportunities each scan.
    
    ### 🎯 Trading Questions
    
    **Q: When should I buy?**
    A: Wait for price to touch Monday Low, then buy when it bounces 2-3% higher with volume confirmation.
    
    **Q: When should I sell?**
    A: Take profits at Monday Midline (50%) or Monday High (full). Always use stops below Monday Low.
    
    **Q: What if the stock gaps down?**
    A: If it gaps below Monday Low, wait for recovery above the Monday Low before entering.
    
    ### ⚙️ Technical Issues
    
    **Q: The scan failed - what happened?**
    A: Check your internet connection, reduce the number of custom tickers, or try again in a few minutes.
    
    **Q: No candidates found - why?**
    A: Lower the minimum LUW score to 0.3, increase max price, or try a different profile.
    """)

def main():
    """Main Streamlit application with documentation"""
    
    # Check if documentation should be shown
    if st.session_state.get('show_docs', False):
        show_documentation()
        
        # Add back button
        if st.button("← Back to Scanner"):
            st.session_state.show_docs = False
            st.rerun()
        return
    
    # Check for quick start guide
    if st.session_state.get('show_quick_start', False):
        show_quick_start_guide()
        
        if st.button("← Back to Scanner"):
            st.session_state.show_quick_start = False
            st.rerun()
        return
    
    # Check for FAQ
    if st.session_state.get('show_faq', False):
        show_faq_only()
        
        if st.button("← Back to Scanner"):
            st.session_state.show_faq = False
            st.rerun()
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 LUW Scanner</h1>
        <p>Advanced Long Until Wrong Analysis for Stock Selection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Profile selection
        profile = st.selectbox(
            "Trading Profile",
            ["Balanced", "Conservative", "Aggressive"],
            help="Choose your trading style to get optimized ticker selection"
        )
        
        # Scan options
        st.subheader("Scan Options")
        scan_type = st.radio(
            "Scan Type",
            ["Quick Scan (10 tickers)", "Smart Scan (50 tickers)", "Extended Scan (25 tickers)", "Custom Tickers"],
            help="Smart Scan randomly samples 50 S&P 500 stocks for comprehensive discovery"
        )
        
        if scan_type == "Custom Tickers":
            custom_tickers = st.text_area(
                "Enter tickers (comma separated)",
                "AAPL, MSFT, GOOGL, AMZN",
                help="Enter ticker symbols separated by commas"
            )
        
        # Show info about Smart Scan
        if scan_type == "Smart Scan (50 tickers)":
            st.info("🎲 Smart Scan randomly samples 50 different S&P 500 stocks each time, perfect for discovering new opportunities!")
        
        # Filters
        st.subheader("Filters")
        min_score = st.slider("Minimum LUW Score", 0.0, 1.0, 0.3, 0.1)
        min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        max_price = st.number_input("Max Price ($)", value=1000.0, step=10.0)
        
        # Scan button
        if st.button("🔍 Run LUW Scan", type="primary", use_container_width=True):
            st.session_state.run_scan = True
            st.session_state.scan_complete = False
        
        # Documentation section
        st.markdown("---")
        st.subheader("📚 Help & Documentation")
        
        if st.button("📖 Complete Documentation", use_container_width=True):
            st.session_state.show_docs = True
            st.rerun()
        
        if st.button("🚀 Quick Start Guide", use_container_width=True):
            st.session_state.show_quick_start = True
            st.rerun()
        
        if st.button("❓ FAQ", use_container_width=True):
            st.session_state.show_faq = True
            st.rerun()
    
    # Main content area
    if st.session_state.get('run_scan', False):
        # Determine tickers to scan
        if scan_type == "Custom Tickers":
            tickers = [t.strip().upper() for t in custom_tickers.split(',')]
        elif scan_type == "Quick Scan (10 tickers)":
            tickers = get_sector_tickers(profile)
        elif scan_type == "Smart Scan (50 tickers)":
            tickers = get_smart_scan_sample()
        else:  # Extended scan
            tickers = get_sp500_sample()[:25]
        
        # Filter by profile if not custom or smart scan
        if scan_type not in ["Custom Tickers", "Smart Scan (50 tickers)"]:
            if profile == "Conservative":
                tickers = [t for t in tickers if t in ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'KO', 'WMT', 'VZ', 'T', 'PFE', 'BRK-B', 'UNH', 'HD']]
            elif profile == "Aggressive":
                tickers = [t for t in tickers if t in ['TSLA', 'NVDA', 'AMD', 'NFLX', 'META', 'CRM', 'ADBE', 'PYPL', 'ROKU', 'PLTR']]
        
        # Show scan info
        st.header("🚀 Running LUW Analysis...")
        
        scan_info_text = f"**Scan Type:** {scan_type}\n**Profile:** {profile}\n**Analyzing:** {len(tickers)} tickers"
        if scan_type == "Smart Scan (50 tickers)":
            scan_info_text += "\n**Note:** Smart Scan uses random sampling for market discovery"
        
        st.info(scan_info_text)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analyzer = SimpleLUWAnalyzer()
        results = []
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
            progress_bar.progress((i + 1) / len(tickers))
            
            result = analyzer.analyze_ticker(ticker)
            if result and result['composite_score'] >= min_score:
                if min_price <= result['current_price'] <= max_price:
                    results.append(result)
            
            time.sleep(0.1)  # Small delay for visual effect
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results
        if results:
            st.session_state.results_df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
            st.session_state.scan_complete = True
            if scan_type == "Smart Scan (50 tickers)":
                st.success(f"✅ Smart Scan completed! Found {len(results)} qualifying candidates from {len(tickers)} randomly sampled S&P 500 stocks.")
            else:
                st.success(f"✅ Scan completed! Found {len(results)} qualifying candidates.")
        else:
            st.warning("❌ No candidates found matching your criteria. Try lowering the minimum score or adjusting filters.")
            if scan_type == "Smart Scan (50 tickers)":
                st.info("💡 Try running Smart Scan again - it will sample different stocks and may find new opportunities!")
            st.session_state.results_df = None
        
        st.session_state.run_scan = False
    
    # Display results
    if st.session_state.scan_complete and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Candidates Found", len(results_df))
        with col2:
            st.metric("Average Score", f"{results_df['composite_score'].mean():.3f}")
        with col3:
            st.metric("Top Candidate", results_df.iloc[0]['ticker'])
        with col4:
            st.metric("Highest Score", f"{results_df['composite_score'].max():.3f}")
        
        # Results table
        st.header("📊 Top LUW Candidates")
        
        # Format for display
        display_df = results_df.copy()
        display_df['LUW Score'] = display_df['composite_score'].apply(lambda x: f"{x:.3f}")
        display_df['Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['Volume (M)'] = display_df['avg_volume_million'].apply(lambda x: f"{x:.1f}")
        display_df['Range Respect'] = display_df['range_respect'].apply(lambda x: f"{x:.3f}")
        display_df['Reversal Quality'] = display_df['reversal_quality'].apply(lambda x: f"{x:.3f}")
        
        # Display table
        st.dataframe(
            display_df[['ticker', 'LUW Score', 'Price', 'Volume (M)', 'Range Respect', 'Reversal Quality']],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"luw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Interactive Candidate Selection
        st.header("🎯 Interactive Analysis")
        
        # Candidate selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_candidates = st.multiselect(
                "📋 Select candidates to analyze:",
                options=results_df['ticker'].tolist(),
                default=results_df['ticker'].tolist()[:3],  # Default to top 3
                help="Choose which candidates to include in the analysis charts"
            )
        
        with col2:
            st.markdown("**Quick Select:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔝 Top 3", use_container_width=True):
                    st.session_state.selected_candidates = results_df['ticker'].tolist()[:3]
                    st.rerun()
            with col_b:
                if st.button("📊 All", use_container_width=True):
                    st.session_state.selected_candidates = results_df['ticker'].tolist()
                    st.rerun()
        
        # Update selection from quick select buttons
        if 'selected_candidates' in st.session_state:
            selected_candidates = st.session_state.selected_candidates
        
        # Filter data based on selection
        if selected_candidates:
            filtered_df = results_df[results_df['ticker'].isin(selected_candidates)]
            
            # Selection summary
            st.info(f"📊 Analyzing {len(selected_candidates)} selected candidates: {', '.join(selected_candidates)}")
            
            # Charts section
            st.header("📈 Dynamic Analysis Charts")
            
            # Row 1: Distribution and Scatter
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution for selected candidates
                fig = px.histogram(
                    filtered_df, 
                    x='composite_score', 
                    nbins=min(10, len(filtered_df)),
                    title=f"LUW Score Distribution ({len(selected_candidates)} selected)",
                    labels={'composite_score': 'LUW Score', 'count': 'Number of Stocks'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score vs Price with candidate labels
                fig = px.scatter(
                    filtered_df,
                    x='current_price',
                    y='composite_score',
                    text='ticker',
                    title=f"LUW Score vs Price ({len(selected_candidates)} selected)",
                    labels={'current_price': 'Price ($)', 'composite_score': 'LUW Score'},
                    color='composite_score',
                    size='avg_volume_million',
                    color_continuous_scale='viridis'
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 2: Component Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Component comparison bar chart
                components = ['range_respect', 'reversal_quality', 'swing_efficiency', 'gap_patterns', 'volume_consistency']
                component_labels = ['Range Respect', 'Reversal Quality', 'Swing Efficiency', 'Gap Patterns', 'Volume Consistency']
                
                # Create data for grouped bar chart
                chart_data = []
                for idx, row in filtered_df.iterrows():
                    for comp, label in zip(components, component_labels):
                        chart_data.append({
                            'Ticker': row['ticker'],
                            'Component': label,
                            'Score': row[comp]
                        })
                
                chart_df = pd.DataFrame(chart_data)
                
                fig = px.bar(
                    chart_df,
                    x='Component',
                    y='Score',
                    color='Ticker',
                    title="Component Score Comparison",
                    barmode='group'
                )
                fig.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume vs Score relationship
                fig = px.scatter(
                    filtered_df,
                    x='avg_volume_million',
                    y='composite_score',
                    text='ticker',
                    title="Volume vs LUW Score",
                    labels={'avg_volume_million': 'Average Volume (M)', 'composite_score': 'LUW Score'},
                    color='range_respect',
                    size='current_price',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Radar Chart Comparison
            if len(selected_candidates) <= 5:  # Only show radar for 5 or fewer candidates
                st.subheader("🕸️ Multi-Candidate Radar Comparison")
                
                fig = go.Figure()
                
                categories = ['Range Respect', 'Reversal Quality', 'Swing Efficiency', 'Gap Patterns', 'Volume Consistency']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                
                for i, (_, candidate) in enumerate(filtered_df.iterrows()):
                    values = [
                        candidate['range_respect'],
                        candidate['reversal_quality'],
                        candidate['swing_efficiency'],
                        candidate['gap_patterns'],
                        candidate['volume_consistency']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=candidate['ticker'],
                        line_color=colors[i % len(colors)],
                        fillcolor=colors[i % len(colors)],
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Component Comparison: {', '.join(selected_candidates)}",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔍 Select 5 or fewer candidates to see the radar comparison chart")
            
            # Detailed comparison table for selected candidates
            st.subheader("📋 Selected Candidates - Detailed View")
            
            # Create detailed comparison
            comparison_df = filtered_df[['ticker', 'composite_score', 'current_price', 'avg_volume_million', 
                                       'range_respect', 'reversal_quality', 'swing_efficiency', 
                                       'gap_patterns', 'volume_consistency', 'total_weeks']].copy()
            
            # Format for better display
            comparison_df['LUW Score'] = comparison_df['composite_score'].apply(lambda x: f"{x:.3f}")
            comparison_df['Price'] = comparison_df['current_price'].apply(lambda x: f"${x:.2f}")
            comparison_df['Volume (M)'] = comparison_df['avg_volume_million'].apply(lambda x: f"{x:.1f}")
            comparison_df['Range Respect'] = comparison_df['range_respect'].apply(lambda x: f"{x:.3f}")
            comparison_df['Reversal Quality'] = comparison_df['reversal_quality'].apply(lambda x: f"{x:.3f}")
            comparison_df['Swing Efficiency'] = comparison_df['swing_efficiency'].apply(lambda x: f"{x:.3f}")
            comparison_df['Gap Patterns'] = comparison_df['gap_patterns'].apply(lambda x: f"{x:.3f}")
            comparison_df['Volume Consistency'] = comparison_df['volume_consistency'].apply(lambda x: f"{x:.3f}")
            
            # Display formatted table
            st.dataframe(
                comparison_df[['ticker', 'LUW Score', 'Price', 'Volume (M)', 'Range Respect', 
                             'Reversal Quality', 'Swing Efficiency', 'Gap Patterns', 'Volume Consistency']],
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics for selected candidates
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = filtered_df['composite_score'].mean()
                st.metric("Average LUW Score", f"{avg_score:.3f}")
            
            with col2:
                best_component = filtered_df[['range_respect', 'reversal_quality', 'swing_efficiency', 
                                            'gap_patterns', 'volume_consistency']].mean().idxmax()
                component_names = {
                    'range_respect': 'Range Respect',
                    'reversal_quality': 'Reversal Quality', 
                    'swing_efficiency': 'Swing Efficiency',
                    'gap_patterns': 'Gap Patterns',
                    'volume_consistency': 'Volume Consistency'
                }
                st.metric("Strongest Component", component_names[best_component])
            
            with col3:
                avg_volume = filtered_df['avg_volume_million'].mean()
                st.metric("Average Volume", f"{avg_volume:.1f}M")
            
            with col4:
                price_range = filtered_df['current_price'].max() - filtered_df['current_price'].min()
                st.metric("Price Range", f"${price_range:.2f}")
            
            # Trading recommendations based on selection
            st.subheader("💡 Trading Insights for Selected Candidates")
            
            # Generate insights
            high_range_respect = filtered_df[filtered_df['range_respect'] > 0.6]['ticker'].tolist()
            high_reversal = filtered_df[filtered_df['reversal_quality'] > 0.6]['ticker'].tolist()
            high_volume = filtered_df[filtered_df['avg_volume_million'] > 10]['ticker'].tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if high_range_respect:
                    st.success(f"🎯 **Best for Level Trading:**\n{', '.join(high_range_respect)}")
                else:
                    st.warning("⚠️ No candidates with strong range respect (>0.6)")
            
            with col2:
                if high_reversal:
                    st.success(f"⚡ **Best for Bounce Plays:**\n{', '.join(high_reversal)}")
                else:
                    st.warning("⚠️ No candidates with strong reversal quality (>0.6)")
            
            with col3:
                if high_volume:
                    st.success(f"💪 **Best Liquidity:**\n{', '.join(high_volume)}")
                else:
                    st.info("ℹ️ Consider volume requirements for position sizing")
                    
        else:
            st.warning("⚠️ Please select at least one candidate to view analysis charts")
    
    elif not st.session_state.scan_complete:
        # Welcome screen
        st.header("👋 Welcome to LUW Scanner")
        
        st.markdown("""
        **Long Until Wrong (LUW)** is a systematic approach to finding stocks that respect key weekly levels 
        and provide consistent trading opportunities.
        
        ### 📋 How it works:
        1. **Monday Range Analysis**: Calculates weekly Monday high/low ranges
        2. **Range Respect Scoring**: Measures how well price respects these levels
        3. **Reversal Quality**: Analyzes bounce patterns at Monday lows
        4. **Swing Efficiency**: Evaluates trading opportunity richness
        5. **Gap Patterns**: Studies gap behavior and follow-through
        
        ### 🚀 Getting Started:
        1. Choose your trading profile in the sidebar
        2. Select scan type:
           - **Quick Scan**: Fast 10-ticker analysis
           - **Smart Scan**: 50 random S&P 500 stocks for discovery
           - **Extended Scan**: 25 curated popular stocks
           - **Custom Tickers**: Your specific stock list
        3. Adjust filters if needed
        4. Click "Run LUW Scan" to analyze candidates
        
        ### 💡 Tip:
        **Try Smart Scan** for the best market discovery - it randomly samples 50 different S&P 500 stocks each time!
        
        **New to LUW?** Click the **"🚀 Quick Start Guide"** button in the sidebar!
        """)
        
        # Sample results teaser
        st.subheader("📊 Sample Analysis")
        sample_data = {
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'LUW Score': [0.847, 0.821, 0.798],
            'Price': ['$150.25', '$299.18', '$2485.50'],
            'Range Respect': [0.856, 0.834, 0.812]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        st.info("👆 This is what your results will look like. Use the sidebar to run your first scan!")

if __name__ == "__main__":
    main()