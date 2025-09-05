# requirements.txt
"""
# LUW Scanner Requirements
# Install with: pip install -r requirements.txt

yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
requests>=2.28.0
beautifulsoup4>=4.11.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.2.0
"""

# example_usage.py
"""
LUW Scanner - Practical Usage Examples
=====================================

This script demonstrates various ways to use the LUW scanner
for different trading scenarios and requirements.
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from luw_scanner_main import LUWScanner, LUWConfig
from luw_config_advanced import AdvancedLUWConfig, ResultsAnalyzer, create_watchlist_export
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_scan():
    """Example 1: Basic LUW scan with default settings"""
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic LUW Scan")
    print("="*60)
    
    # Create scanner with default configuration
    scanner = LUWScanner()
    
    # For demonstration, let's scan just a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']
    
    print(f"Scanning {len(test_tickers)} test tickers...")
    
    results = []
    for ticker in test_tickers:
        print(f"Analyzing {ticker}...")
        result = scanner.analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        print("\nResults:")
        print(results_df[['ticker', 'composite_score', 'current_price', 'range_respect', 'reversal_quality']].to_string(index=False))
        
        return results_df
    else:
        print("No results found.")
        return pd.DataFrame()

def example_2_conservative_profile():
    """Example 2: Conservative trading profile"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Conservative Profile Scan")
    print("="*60)
    
    # Use conservative configuration
    config = AdvancedLUWConfig.get_conservative_config()
    scanner = LUWScanner(config)
    
    print("Configuration:")
    print(f"  Min Volume: {config.min_volume_million}M")
    print(f"  Min Price: ${config.min_price}")
    print(f"  Range Tolerance: {config.range_tolerance_pct}%")
    print(f"  Min Respect Weeks: {config.min_respect_weeks}")
    
    # Scan high-quality large caps
    large_cap_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'JPM', 'PG']
    
    results = []
    for ticker in large_cap_tickers:
        print(f"Analyzing {ticker} (conservative)...")
        result = scanner.analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        print("\nConservative Profile Results:")
        print(results_df[['ticker', 'composite_score', 'range_respect']].to_string(index=False))
        
        return results_df
    else:
        print("No conservative candidates found.")
        return pd.DataFrame()

def example_3_aggressive_profile():
    """Example 3: Aggressive trading profile for active traders"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Aggressive Profile Scan")
    print("="*60)
    
    # Use aggressive configuration
    config = AdvancedLUWConfig.get_aggressive_config()
    scanner = LUWScanner(config)
    
    print("Configuration:")
    print(f"  Min Volume: {config.min_volume_million}M")
    print(f"  Min Price: ${config.min_price}")
    print(f"  Range Tolerance: {config.range_tolerance_pct}%")
    print(f"  Swing Efficiency Weight: {config.weights['swing_efficiency']}")
    
    # Scan more volatile growth stocks
    growth_tickers = ['TSLA', 'NVDA', 'AMD', 'NFLX', 'SHOP', 'ROKU', 'PLTR', 'SNOW']
    
    results = []
    for ticker in growth_tickers:
        print(f"Analyzing {ticker} (aggressive)...")
        result = scanner.analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        print("\nAggressive Profile Results:")
        print(results_df[['ticker', 'composite_score', 'swing_efficiency', 'gap_patterns']].to_string(index=False))
        
        return results_df
    else:
        print("No aggressive candidates found.")
        return pd.DataFrame()

def example_4_custom_configuration():
    """Example 4: Custom configuration for specific requirements"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    custom_config = LUWConfig(
        lookback_weeks=39,  # 9 months
        min_volume_million=3.0,  # Mid-high volume
        min_price=15.0,  # Mid-range price
        range_tolerance_pct=1.8,  # Moderate tolerance
        min_respect_weeks=25,  # Good respect required
        reversal_candle_limit=5,  # Quick reversals
        weights={
            'range_respect': 0.25,
            'reversal_quality': 0.35,  # Emphasis on reversals
            'swing_efficiency': 0.25,
            'gap_patterns': 0.10,
            'volume_consistency': 0.05
        }
    )
    
    scanner = LUWScanner(custom_config)
    
    print("Custom configuration emphasizes reversal quality")
    
    # Scan mid-cap stocks
    mid_cap_tickers = ['CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN']
    
    results = []
    for ticker in mid_cap_tickers:
        print(f"Analyzing {ticker} (custom)...")
        result = scanner.analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        print("\nCustom Configuration Results:")
        print(results_df[['ticker', 'composite_score', 'reversal_quality']].to_string(index=False))
        
        return results_df
    else:
        print("No custom candidates found.")
        return pd.DataFrame()

def example_5_sector_comparison():
    """Example 5: Compare different sectors"""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Sector Comparison")
    print("="*60)
    
    scanner = LUWScanner()
    
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD']
    }
    
    sector_results = {}
    
    for sector_name, tickers in sectors.items():
        print(f"\nAnalyzing {sector_name} sector...")
        sector_scores = []
        
        for ticker in tickers:
            result = scanner.analyze_ticker(ticker)
            if result:
                sector_scores.append(result['composite_score'])
        
        if sector_scores:
            sector_results[sector_name] = {
                'avg_score': sum(sector_scores) / len(sector_scores),
                'max_score': max(sector_scores),
                'count': len(sector_scores)
            }
    
    print("\nSector Comparison:")
    print("-" * 40)
    for sector, stats in sorted(sector_results.items(), key=lambda x: x[1]['avg_score'], reverse=True):
        print(f"{sector:12s} | Avg: {stats['avg_score']:.3f} | Max: {stats['max_score']:.3f} | Count: {stats['count']}")

def example_6_advanced_analysis():
    """Example 6: Advanced analysis with visualization"""
    
    print("\n" + "="*60)
    print("EXAMPLE 6: Advanced Analysis")
    print("="*60)
    
    # Run a broader scan first
    scanner = LUWScanner()
    
    # Scan broader universe
    broad_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
                    'CRM', 'ADBE', 'JPM', 'JNJ', 'PG', 'KO', 'DIS', 'BA']
    
    print(f"Running broader scan on {len(broad_tickers)} tickers...")
    
    results = []
    for ticker in broad_tickers:
        result = scanner.analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if not results:
        print("No results for advanced analysis.")
        return
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('composite_score', ascending=False)
    
    # Perform advanced analysis
    analyzer = ResultsAnalyzer(results_df)
    
    # Generate summary statistics
    summary = analyzer.generate_summary_stats()
    print(f"\nAnalyzed {summary.get('total_candidates', 0)} candidates")
    print(f"Top 10% threshold: {summary.get('top_10_percent_threshold', 0):.3f}")
    print(f"Average composite score: {summary.get('composite_score', {}).get('mean', 0):.3f}")
    
    # Identify outliers
    outliers = analyzer.identify_outliers()
    if not outliers.empty:
        print(f"\nFound {len(outliers)} outlier candidates:")
        print(outliers[['ticker', 'composite_score']].head().to_string(index=False))
    
    # Create exports
    try:
        # Export top 10 as watchlist
        watchlist_file = create_watchlist_export(results_df, top_n=10, export_format='csv')
        print(f"\nWatchlist exported to: {watchlist_file}")
        
        # Create visualizations (if matplotlib available)
        print("Generating visualizations...")
        analyzer.plot_score_distribution("luw_score_distribution.png")
        analyzer.plot_correlation_heatmap("luw_correlation_heatmap.png")
        print("Visualizations saved as PNG files")
        
        # Create interactive dashboard
        analyzer.create_interactive_dashboard("luw_dashboard.html")
        print("Interactive dashboard saved as luw_dashboard.html")
        
    except Exception as e:
        print(f"Visualization error (likely missing dependencies): {e}")
    
    return results_df

def example_7_quick_single_ticker():
    """Example 7: Quick analysis of a single ticker"""
    
    print("\n" + "="*60)
    print("EXAMPLE 7: Single Ticker Deep Dive")
    print("="*60)
    
    ticker = "AAPL"  # You can change this to any ticker
    
    scanner = LUWScanner()
    
    print(f"Performing deep analysis on {ticker}...")
    
    # Get the raw data first
    data = scanner.data_manager.get_ticker_data(ticker)
    if not data:
        print(f"Could not retrieve data for {ticker}")
        return
    
    daily_data = data['daily']
    intraday_data = data['intraday']
    
    print(f"Daily data: {len(daily_data)} days")
    print(f"Intraday data: {len(intraday_data)} 15-min candles")
    print(f"Current price: ${daily_data['Close'].iloc[-1]:.2f}")
    print(f"Average volume: {daily_data['Volume'].mean() / 1_000_000:.1f}M")
    
    # Calculate Monday ranges
    monday_ranges = scanner.monday_calculator.calculate_monday_ranges(daily_data)
    print(f"Monday ranges calculated: {len(monday_ranges)} weeks")
    
    if not monday_ranges.empty:
        latest_range = monday_ranges.iloc[-1]
        print(f"\nLatest Monday Range:")
        print(f"  High: ${latest_range['monday_high']:.2f}")
        print(f"  Mid:  ${latest_range['monday_mid']:.2f}")
        print(f"  Low:  ${latest_range['monday_low']:.2f}")
        print(f"  Range: {latest_range['range_pct']:.1f}%")
    
    # Perform full analysis
    result = scanner.analyze_ticker(ticker)
    if result:
        print(f"\nLUW Analysis Results for {ticker}:")
        print("-" * 40)
        print(f"Composite Score:    {result['composite_score']:.3f}")
        print(f"Range Respect:      {result['range_respect']:.3f}")
        print(f"Reversal Quality:   {result['reversal_quality']:.3f}")
        print(f"Swing Efficiency:   {result['swing_efficiency']:.3f}")
        print(f"Gap Patterns:       {result['gap_patterns']:.3f}")
        
        # Provide interpretation
        score = result['composite_score']
        if score > 0.8:
            interpretation = "EXCELLENT LUW candidate"
        elif score > 0.6:
            interpretation = "GOOD LUW candidate"
        elif score > 0.4:
            interpretation = "AVERAGE LUW candidate"
        else:
            interpretation = "POOR LUW candidate"
        
        print(f"\nInterpretation: {interpretation}")
    
    return result

def main():
    """Run all examples"""
    
    print("LUW SCANNER - USAGE EXAMPLES")
    print("="*80)
    print("This script demonstrates various ways to use the LUW scanner.")
    print("Each example shows different configurations and use cases.")
    print("\nNote: These examples use limited tickers for demonstration.")
    print("For production use, run the full S&P 500 scan with scanner.scan_all_sp500()")
    
    # Run examples
    try:
        example_1_basic_scan()
        example_2_conservative_profile()
        example_3_aggressive_profile()
        example_4_custom_configuration()
        example_5_sector_comparison()
        example_7_quick_single_ticker()
        example_6_advanced_analysis()  # Run last as it's most comprehensive
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install yfinance pandas numpy requests beautifulsoup4 matplotlib seaborn plotly")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("1. Install required packages if you haven't: pip install -r requirements.txt")
    print("2. Run full S&P 500 scan: scanner.scan_all_sp500()")
    print("3. Customize configuration for your trading style")
    print("4. Use the advanced analysis tools for deeper insights")
    print("5. Export watchlists and set up alerts for top candidates")

if __name__ == "__main__":
    main()

# Installation and Setup Instructions
"""
LUW Scanner Installation & Setup
===============================

1. INSTALL REQUIREMENTS:
   pip install yfinance pandas numpy requests beautifulsoup4 matplotlib seaborn plotly scikit-learn

2. BASIC USAGE:
   from luw_scanner_main import LUWScanner
   scanner = LUWScanner()
   results = scanner.scan_all_sp500()

3. CONFIGURATION:
   from luw_config_advanced import AdvancedLUWConfig
   config = AdvancedLUWConfig.get_conservative_config()
   scanner = LUWScanner(config)

4. ANALYSIS:
   from luw_config_advanced import ResultsAnalyzer
   analyzer = ResultsAnalyzer(results)
   analyzer.create_interactive_dashboard()

5. FILE STRUCTURE:
   luw_scanner_main.py         # Main scanner classes
   luw_config_advanced.py      # Advanced configuration and analysis
   example_usage.py            # This file - usage examples
   requirements.txt            # Required packages

6. OUTPUT FILES:
   luw_cache/                  # Cached data directory
   luw_scan_results_*.csv      # Scanner results
   luw_dashboard.html          # Interactive dashboard
   luw_watchlist_*.csv         # Exported watchlists
   *.png                       # Visualization plots

7. CUSTOMIZATION:
   - Modify LUWConfig parameters for different trading styles
   - Adjust scoring weights in the config
   - Add custom filters in _passes_basic_filters()
   - Extend analysis with additional patterns

8. PERFORMANCE TIPS:
   - Use caching (enabled by default)
   - Reduce max_workers for slower connections
   - Filter tickers before scanning for faster results
   - Use smaller lookback periods for quicker analysis

9. TROUBLESHOOTING:
   - Check internet connection for data downloads
   - Verify ticker symbols are correct
   - Clear cache directory if data seems stale
   - Reduce batch size if getting API errors

10. PRODUCTION DEPLOYMENT:
    - Set up scheduled scans (daily/weekly)
    - Implement alerting for new high-scoring candidates
    - Store results in database for historical analysis
    - Create automated reporting and visualization updates
"""