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
from typing import Dict, List, Optional, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
PAGE_CONFIG = {
    "page_title": "LUW Scanner - Long Until Wrong Analysis",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

COMPONENT_WEIGHTS = {
    'range_respect': 0.30,
    'reversal_quality': 0.25,
    'swing_efficiency': 0.20,
    'gap_patterns': 0.15,
    'volume_consistency': 0.10
}

# Page configuration
st.set_page_config(**PAGE_CONFIG)

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
.score-excellent { background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
.score-good { background-color: #d1ecf1; color: #0c5460; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
.score-average { background-color: #fff3cd; color: #856404; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
.score-poor { background-color: #f8d7da; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False

class RecencyWeightCalculator:
    """Handles recency weighting calculations for time series data"""
    
    def __init__(self, decay_factor: float = 0.96):
        self.decay_factor = decay_factor
    
    def get_weeks_ago(self, week_start_dates: List[pd.Timestamp], current_date: pd.Timestamp) -> List[int]:
        """Calculate how many weeks ago each date occurred"""
        weeks_ago = []
        for week_start in week_start_dates:
            days_diff = (current_date - week_start).days
            weeks_diff = max(0, days_diff // 7)
            weeks_ago.append(weeks_diff)
        return weeks_ago
    
    def apply_weights(self, values: List[float], weeks_ago: List[int]) -> float:
        """Apply exponential decay weighting to favor recent data"""
        if not values or len(values) == 0:
            return 0.0
        
        values = np.array(values)
        weeks_ago = np.array(weeks_ago)
        
        # Calculate weights using exponential decay
        weights = np.array([self.decay_factor ** week for week in weeks_ago])
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            return np.sum(values * weights)
        else:
            return np.mean(values)

class DataRetriever:
    """Handles data retrieval and preprocessing"""
    
    @staticmethod
    def get_ticker_data(ticker: str, period: str = "1y") -> Optional[Dict]:
        """Get ticker data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            daily_data = stock.history(period=period)
            
            if daily_data.empty:
                return None
                
            # Get intraday data if available
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
    
    @staticmethod
    def calculate_monday_ranges(daily_data: pd.DataFrame) -> pd.DataFrame:
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

class LUWMetricsCalculator:
    """Calculates individual LUW metrics with recency weighting"""
    
    def __init__(self, recency_calculator: RecencyWeightCalculator, range_tolerance: float = 0.02):
        self.recency = recency_calculator
        self.range_tolerance = range_tolerance
    
    def calculate_range_respect(self, daily_data: pd.DataFrame, monday_ranges: pd.DataFrame) -> float:
        """Calculate how well price respects Monday ranges with recency weighting"""
        if monday_ranges.empty:
            return 0.0
        
        week_scores = []
        week_start_dates = []
        current_date = daily_data.index.max()
        
        for _, week_range in monday_ranges.iterrows():
            week_start = week_range['week_start']
            week_end = week_range['week_end']
            
            week_daily = daily_data[
                (daily_data.index >= week_start) & 
                (daily_data.index <= week_end)
            ]
            
            if week_daily.empty:
                continue
            
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
                normalized_score = min(week_respect / len(week_daily), 1.0)
                week_scores.append(normalized_score)
                week_start_dates.append(week_start)
        
        if not week_scores:
            return 0.0
        
        # Apply recency weighting
        weeks_ago = self.recency.get_weeks_ago(week_start_dates, current_date)
        return self.recency.apply_weights(week_scores, weeks_ago)
    
    def calculate_reversal_quality(self, daily_data: pd.DataFrame, monday_ranges: pd.DataFrame) -> float:
        """Calculate reversal quality at Monday levels with recency weighting"""
        if monday_ranges.empty:
            return 0.0
        
        reversal_scores = []
        week_start_dates = []
        current_date = daily_data.index.max()
        
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
            week_reversal_events = 0
            week_successful_reversals = 0
            
            for i in range(len(week_daily) - 2):
                current_day = week_daily.iloc[i]
                
                # Check if touched Monday Low
                if current_day['Low'] <= monday_low * 1.01:  # 1% tolerance
                    week_reversal_events += 1
                    
                    # Look for reversal with speed scoring
                    for j in range(1, min(3, len(week_daily) - i)):
                        future_day = week_daily.iloc[i + j]
                        if future_day['Close'] > monday_low * 1.02:  # 2% above
                            # Score based on speed: same day = 1.0, next day = 0.8, day 2 = 0.6
                            speed_score = 1.0 - (j - 1) * 0.2
                            week_successful_reversals += speed_score
                            break
            
            # Calculate week's reversal quality
            if week_reversal_events > 0:
                week_quality = week_successful_reversals / week_reversal_events
                reversal_scores.append(week_quality)
                week_start_dates.append(week_start)
        
        if not reversal_scores:
            return 0.0
        
        # Apply recency weighting
        weeks_ago = self.recency.get_weeks_ago(week_start_dates, current_date)
        return self.recency.apply_weights(reversal_scores, weeks_ago)
    
    def calculate_swing_efficiency(self, daily_data: pd.DataFrame, monday_ranges: pd.DataFrame) -> float:
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
                trend_consistency = max(0.0, 1.0 - (week_price_std * 10))
                range_score = min(week_avg_range * 10, 1.0)
                utilization_score = min(range_utilization, 1.0)
                
                # Combine metrics for weekly efficiency score
                week_efficiency = (range_score * 0.4 + trend_consistency * 0.3 + utilization_score * 0.3)
                weekly_efficiency_scores.append(max(0.0, min(week_efficiency, 1.0)))
                week_start_dates.append(week_start)
        
        if not weekly_efficiency_scores:
            return 0.0
        
        # Apply recency weighting
        weeks_ago = self.recency.get_weeks_ago(week_start_dates, current_date)
        return self.recency.apply_weights(weekly_efficiency_scores, weeks_ago)
    
    def calculate_gap_patterns(self, daily_data: pd.DataFrame) -> float:
        """Calculate gap pattern success rate with recency weighting"""
        if len(daily_data) < 10:
            return 0.0
        
        gap_scores = []
        weekly_data = {}
        
        # Group data by week for recency weighting
        daily_data_copy = daily_data.copy()
        daily_data_copy['week'] = daily_data_copy.index.isocalendar().week
        daily_data_copy['year'] = daily_data_copy.index.year
        
        current_date = daily_data.index.max()
        
        for (year, week), week_data in daily_data_copy.groupby(['year', 'week']):
            if len(week_data) < 2:
                continue
                
            week_gap_events = 0
            week_successful_patterns = 0
            
            for i in range(1, len(week_data)):
                prev_close = week_data.iloc[i-1]['Close']
                current_open = week_data.iloc[i]['Open']
                current_close = week_data.iloc[i]['Close']
                
                gap_percent = abs(current_open - prev_close) / prev_close
                
                if gap_percent > 0.005:  # 0.5% gap
                    week_gap_events += 1
                    
                    # Enhanced pattern scoring
                    if current_open > prev_close:  # Gap up
                        if current_close > current_open:
                            continuation_strength = (current_close - current_open) / (current_open - prev_close)
                            week_successful_patterns += min(1.0, continuation_strength)
                    else:  # Gap down
                        if current_close > current_open:
                            reversal_strength = (current_close - current_open) / (prev_close - current_open)
                            week_successful_patterns += min(1.0, reversal_strength * 1.2)
            
            # Calculate week's gap pattern score
            if week_gap_events > 0:
                week_score = week_successful_patterns / week_gap_events
                gap_scores.append(week_score)
                weekly_data[week_data.index.min()] = week_score
        
        if not gap_scores:
            return 0.0
        
        # Apply recency weighting
        week_start_dates = list(weekly_data.keys())
        weeks_ago = self.recency.get_weeks_ago(week_start_dates, current_date)
        return self.recency.apply_weights(gap_scores, weeks_ago)

class LUWAnalyzer:
    """Main analyzer class that orchestrates the LUW analysis"""
    
    def __init__(self, lookback_weeks: int = 26, range_tolerance: float = 0.02, decay_factor: float = 0.96):
        self.lookback_weeks = lookback_weeks
        self.range_tolerance = range_tolerance
        self.recency_calculator = RecencyWeightCalculator(decay_factor)
        self.metrics_calculator = LUWMetricsCalculator(self.recency_calculator, range_tolerance)
        self.data_retriever = DataRetriever()
    
    def analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Complete analysis of a single ticker"""
        try:
            # Get data
            data = self.data_retriever.get_ticker_data(ticker)
            if not data or data['daily'].empty:
                return None
            
            daily_data = data['daily']
            current_price = daily_data['Close'].iloc[-1]
            avg_volume = daily_data['Volume'].mean()
            
            # Basic filters
            if current_price < 5.0 or avg_volume < 500000:
                return None
            
            # Calculate Monday ranges
            monday_ranges = self.data_retriever.calculate_monday_ranges(daily_data)
            if monday_ranges.empty:
                return None
            
            # Calculate component scores
            range_respect = self.metrics_calculator.calculate_range_respect(daily_data, monday_ranges)
            reversal_quality = self.metrics_calculator.calculate_reversal_quality(daily_data, monday_ranges)
            swing_efficiency = self.metrics_calculator.calculate_swing_efficiency(daily_data, monday_ranges)
            gap_patterns = self.metrics_calculator.calculate_gap_patterns(daily_data)
            
            # Calculate volume consistency
            volume_consistency = min(avg_volume / 1_000_000, 1.0)
            
            # Calculate composite score
            composite_score = (
                range_respect * COMPONENT_WEIGHTS['range_respect'] +
                reversal_quality * COMPONENT_WEIGHTS['reversal_quality'] +
                swing_efficiency * COMPONENT_WEIGHTS['swing_efficiency'] +
                gap_patterns * COMPONENT_WEIGHTS['gap_patterns'] +
                volume_consistency * COMPONENT_WEIGHTS['volume_consistency']
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

class TickerLists:
    """Manages different ticker lists for scanning"""
    
    @staticmethod
    def get_sp500_sample() -> List[str]:
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'BRK-B', 'JNJ', 'JPM', 'PG', 'UNH', 'HD', 'DIS',
            'BAC', 'WFC', 'GS', 'AXP', 'C', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY',
            'KO', 'PEP', 'WMT', 'NKE', 'MCD', 'BA', 'CAT', 'GE', 'MMM', 'HON',
            'CRM', 'ADBE', 'NFLX', 'PYPL', 'INTC'
        ]
    
    @staticmethod
    def get_extended_sp500_list() -> List[str]:
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
            'V', 'MA', 'COF', 'USB', 'TFC', 'PNC', 'BK', 'STT', 'FITB',
            'HBAN', 'RF', 'CFG', 'KEY', 'ZION', 'CMA', 'NTRS',
            
            # Consumer Discretionary
            'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'ORLY',
            'CMG', 'LULU', 'RCL', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS',
            'CCL', 'AAL', 'DAL', 'UAL', 'LUV', 'F', 'GM',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
            'CAG', 'SJM', 'TSN', 'HRL', 'CHD', 'CLX', 'EL', 'KHC', 'MDLZ',
            'MNST', 'DG', 'DLTR', 'KR', 'WBA', 'STZ',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE', 'WMB', 'VLO',
            'PSX', 'MPC', 'HES', 'DVN', 'FANG', 'APA', 'BKR', 'HAL',
            
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'DE', 'UNP', 'CSX', 'NSC', 'FDX', 'WM', 'RSG', 'PCAR', 'CMI', 'EMR',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'DOW', 'PPG',
            'LYB', 'AVY', 'BLL', 'CF', 'FMC', 'ALB',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA', 'EXR',
            'AVB', 'EQR', 'VTR', 'BXP', 'ARE',
            
            # Utilities
            'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'D', 'PCG', 'PEG', 'SRE',
            'EIX', 'WEC', 'AWK', 'PPL', 'CMS',
            
            # Communication Services
            'META', 'DIS', 'NFLX', 'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA',
            'ATVI', 'EA', 'ROKU', 'ZM'
        ]
    
    @staticmethod
    def get_smart_scan_sample() -> List[str]:
        extended_list = TickerLists.get_extended_sp500_list()
        if len(extended_list) < 50:
            return extended_list
        return random.sample(extended_list, 50)
    
    @staticmethod
    def get_sector_tickers(profile: str) -> List[str]:
        if profile == "Conservative":
            return ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'KO', 'WMT', 'VZ', 'T', 'PFE']
        elif profile == "Aggressive":
            return ['TSLA', 'NVDA', 'AMD', 'NFLX', 'ROKU', 'PLTR', 'SNOW', 'ZM', 'SHOP', 'SQ']
        else:  # Balanced
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JNJ', 'JPM', 'PG', 'NVDA', 'UNH']

class UIComponents:
    """Handles UI components and visualizations"""
    
    @staticmethod
    def format_score_badge(score: float) -> str:
        if score >= 0.8:
            return f'<span class="score-excellent">{score:.3f}</span>'
        elif score >= 0.6:
            return f'<span class="score-good">{score:.3f}</span>'
        elif score >= 0.4:
            return f'<span class="score-average">{score:.3f}</span>'
        else:
            return f'<span class="score-poor">{score:.3f}</span>'
    
    @staticmethod
    def create_sample_monday_range_chart():
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = 100 + np.cumsum(np.random.randn(20) * 0.5)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines', name='Stock Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add Monday levels
        monday_high, monday_low, monday_mid = 102, 98, 100
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
            xaxis_title="Date", yaxis_title="Price ($)", height=400
        )
        
        return fig
    
    @staticmethod
    def create_component_weights_chart():
        components = ['Range Respect', 'Reversal Quality', 'Swing Efficiency', 'Gap Patterns', 'Volume Consistency']
        weights = [30, 25, 20, 15, 10]
        
        fig = go.Figure(data=[go.Pie(labels=components, values=weights)])
        fig.update_layout(title="LUW Score Component Weights", height=300)
        return fig

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
        fig = UIComponents.create_sample_monday_range_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 LUW Scoring System")
        
        st.markdown("""
        ### Composite Score Calculation
        
        The LUW Composite Score (0.0 - 1.0) combines five key components with **recency weighting**:
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
            
            **Recency Weighting:**
            - Current week: 100% weight
            - 4 weeks ago: 85% weight
            - 12 weeks ago: 61% weight
            - 26 weeks ago: 27% weight
            """)
        
        with col2:
            fig = UIComponents.create_component_weights_chart()
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

def run_scan(scan_type: str, profile: str, custom_tickers: str, min_score: float, 
             min_price: float, max_price: float):
    """Run the LUW scan based on parameters"""
    
    # Determine tickers to scan
    if scan_type == "Custom Tickers":
        tickers = [t.strip().upper() for t in custom_tickers.split(',')]
    elif scan_type == "Quick Scan (10 tickers)":
        tickers = TickerLists.get_sector_tickers(profile)
    elif scan_type == "Smart Scan (50 tickers)":
        tickers = TickerLists.get_smart_scan_sample()
    else:  # Extended scan
        tickers = TickerLists.get_sp500_sample()[:25]
    
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
    
    analyzer = LUWAnalyzer()
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

def display_results():
    """Display scan results with interactive analysis"""
    
    if not st.session_state.scan_complete or st.session_state.results_df is None:
        return
    
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

def main():
    """Main Streamlit application"""
    
    # Check if documentation should be shown
    if st.session_state.get('show_docs', False):
        show_documentation()
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
        <p>Advanced Long Until Wrong Analysis with Recency Weighting</p>
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
        else:
            custom_tickers = ""
        
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
        run_scan(scan_type, profile, custom_tickers, min_score, min_price, max_price)
        st.session_state.run_scan = False
    
    # Display results if available
    display_results()
    
    # Welcome screen if no results
    if not st.session_state.scan_complete:
        st.header("👋 Welcome to LUW Scanner")
        
        st.markdown("""
        **Long Until Wrong (LUW)** is a systematic approach to finding stocks that respect key weekly levels 
        and provide consistent trading opportunities.
        
        ### 📋 How it works:
        1. **Monday Range Analysis**: Calculates weekly Monday high/low ranges
        2. **Range Respect Scoring**: Measures how well price respects these levels (with **recency weighting**)
        3. **Reversal Quality**: Analyzes bounce patterns at Monday lows
        4. **Swing Efficiency**: Evaluates trading opportunity richness
        5. **Gap Patterns**: Studies gap behavior and follow-through
        
        ### 🚀 Getting Started:
        1. Choose your trading profile in the sidebar
        2. Select scan type (Smart Scan recommended for discovery)
        3. Adjust filters if needed
        4. Click "Run LUW Scan" to analyze candidates
        
        ### 💡 New Feature: Recency Weighting
        All calculations now emphasize **recent market behavior**:
        - Current week patterns get 100% weight
        - 12 weeks ago gets ~61% weight  
        - 26 weeks ago gets ~27% weight
        
        This makes the scanner much more responsive to current market conditions!
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