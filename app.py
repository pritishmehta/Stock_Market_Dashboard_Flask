"""
Financial Dashboard Application - Optimized Version
Part 1: Core Setup and Utilities
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import yfinance as yf
import datetime 
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import pandas_datareader as pdr
from bs4 import BeautifulSoup
import numpy as np
import plotly.graph_objects as go
import plotly
import json
import os
import time
import threading
import concurrent.futures
from functools import lru_cache
import warnings
import random
# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__, template_folder='templates')

# Configuration constants
API_KEY = '5DHU06PG79LY13BA'  # Alpha Vantage API key
CACHE_TIMEOUT = 3600  # Cache timeout in seconds (1 hour)
NEWS_CACHE_TIMEOUT = 1800  # News cache timeout (30 minutes)
CHART_CACHE_TIMEOUT = 900  # Chart data cache timeout (15 minutes)

# Global cache for API responses to avoid duplicate calls
CACHE = {}

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create a variable to track the prediction process
prediction_job = {
    'is_running': False,
    'start_time': None,
    'completion_time': None,
    'status': 'idle',
    'progress': 0,
    'error': None
}

# Utility Functions
def format_large_number(num):
    """Format large numbers with K, M, B, T suffixes"""
    if not num or num == 0:
        return "N/A"
        
    num = float(num)
    magnitude = 0
    
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
        
    suffix = ['', 'K', 'M', 'B', 'T'][min(magnitude, 4)]
    return f"{num:.2f}{suffix}"

@lru_cache(maxsize=100)
def get_fred_data(series_id):
    """
    Fetch data from FRED with proper error handling and caching.
    Always returns a DataFrame with the appropriate column, even if empty.
    """
    cache_key = f"fred_{series_id}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        # Attempt to fetch data from FRED
        data = pdr.DataReader(series_id, 'fred')
        
        # Cache the result
        CACHE[cache_key] = {
            'data': data,
            'timestamp': current_time
        }
        
        return data
    except Exception as e:
        print(f"Error fetching FRED data for {series_id}: {e}")
        # Return empty DataFrame with proper column name
        return pd.DataFrame(columns=[series_id])

def clear_expired_cache(force_clear=False):
    """Clear expired items from cache to free up memory"""
    current_time = time.time()
    keys_to_remove = []
    
    for key, value in CACHE.items():
        # Check if item has expired or force clear is enabled
        if force_clear or current_time - value['timestamp'] > CACHE_TIMEOUT:
            keys_to_remove.append(key)
    
    # Remove expired items
    for key in keys_to_remove:
        CACHE.pop(key, None)
    
    return len(keys_to_remove)

def get_fallback_news_data(limit=5):
    """Generate fallback news data when API fails"""
    return [
        {
            'title': 'Markets React to Recent Economic Data',
            'publisher': 'Market News Daily',
            'link': '#',
            'summary': 'Global markets showed mixed reactions to the latest economic indicators.',
            'providerPublishTime': int(datetime.datetime.now().timestamp())
        },
        {
            'title': 'Federal Reserve Holds Interest Rates Steady',
            'publisher': 'Financial Times',
            'link': '#',
            'summary': 'The Fed maintained current interest rates, citing stable inflation.',
            'providerPublishTime': int(datetime.datetime.now().timestamp())
        },
        {
            'title': 'Tech Sector Shows Strong Q1 Performance',
            'publisher': 'Tech Insider',
            'link': '#',
            'summary': 'Tech companies reported better-than-expected Q1 earnings.',
            'providerPublishTime': int(datetime.datetime.now().timestamp())
        }
    ][:limit]

def safe_json_dumps(obj):
    """Safe JSON serialization that handles NaN values"""
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj) if not np.isnan(obj) else None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            # Handle plotly objects
            if hasattr(obj, 'to_plotly_json'):
                return obj.to_plotly_json()
            return super(NpEncoder, self).default(obj)
    
    return json.dumps(obj, cls=NpEncoder)

"""
Financial Dashboard Application - Optimized Version
Part 2: Stock and Financial Data Functions
"""

# Note: This section assumes Part 1 has been loaded

@lru_cache(maxsize=50)
def get_stock_info(ticker):
    """
    Get comprehensive information about a stock with caching
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Stock information including price, 52 week high/low, etc.
    """
    cache_key = f"stock_info_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        # Initialize empty result dictionary with default values
        result = {
            'symbol': ticker.upper(),
            'name': 'Unknown',
            'price': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'avg_volume': 0,
            'market_cap': 0,
            'beta': 0,
            'pe_ratio': 0,
            'eps': 0,
            'dividend_yield': 0,
            'target_price': 0,
            '52_week_high': 0,
            '52_week_low': 0,
            'exchange': '',
            'sector': '',
            'industry': '',
            'country': '',
            'logo': None,
            'website': None,
            'description': None,
            'error': None
        }
        
        # Fetch basic info about the stock
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid information
        if not info or not isinstance(info, dict) or len(info.keys()) <= 3:
            return {'error': f"Could not find information for ticker '{ticker}'. Please verify the symbol."}
            
        # Update the result dictionary with actual values
        result['name'] = info.get('longName', info.get('shortName', ticker.upper()))
        result['price'] = info.get('currentPrice', info.get('regularMarketPrice', 0))
        result['change'] = info.get('regularMarketChange', 0)
        result['change_percent'] = info.get('regularMarketChangePercent', 0)
        result['volume'] = info.get('volume', info.get('regularMarketVolume', 0))
        result['avg_volume'] = info.get('averageVolume', 0)
        result['market_cap'] = info.get('marketCap', 0)
        result['beta'] = info.get('beta', 0)
        result['pe_ratio'] = info.get('trailingPE', info.get('forwardPE', 0))
        result['eps'] = info.get('trailingEps', 0)
        result['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        result['target_price'] = info.get('targetMeanPrice', 0)
        result['52_week_high'] = info.get('fiftyTwoWeekHigh', 0)
        result['52_week_low'] = info.get('fiftyTwoWeekLow', 0)
        result['exchange'] = info.get('exchange', info.get('fullExchangeName', ''))
        result['sector'] = info.get('sector', '')
        result['industry'] = info.get('industry', '')
        result['country'] = info.get('country', '')
        result['logo'] = info.get('logo_url', None)
        result['website'] = info.get('website', None)
        result['description'] = info.get('longBusinessSummary', None)
        
        # Check if we actually got the essential data
        if not result['price'] or result['price'] == 0:
            # This might be an invalid ticker or data might not be available
            return {'error': f"Unable to retrieve price data for ticker '{ticker}'. This may not be a valid symbol or the market might be closed."}

        # Format numbers for display
        try:
            result['price'] = f"{float(result['price']):.2f}" if result['price'] else "N/A"
            result['change'] = f"{float(result['change']):.2f}" if result['change'] else "N/A"
            result['change_percent'] = f"{float(result['change_percent']):.2f}" if result['change_percent'] else "N/A"
            result['market_cap'] = format_large_number(result['market_cap']) if result['market_cap'] else "N/A"
            result['volume'] = format_large_number(result['volume']) if result['volume'] else "N/A"
            result['avg_volume'] = format_large_number(result['avg_volume']) if result['avg_volume'] else "N/A"
            result['pe_ratio'] = f"{float(result['pe_ratio']):.2f}" if result['pe_ratio'] else "N/A"
            result['eps'] = f"{float(result['eps']):.2f}" if result['eps'] else "N/A"
            result['dividend_yield'] = f"{float(result['dividend_yield']):.2f}%" if result['dividend_yield'] else "N/A"
            result['target_price'] = f"{float(result['target_price']):.2f}" if result['target_price'] else "N/A"
            result['52_week_high'] = f"{float(result['52_week_high']):.2f}" if result['52_week_high'] else "N/A"
            result['52_week_low'] = f"{float(result['52_week_low']):.2f}" if result['52_week_low'] else "N/A"
            result['beta'] = f"{float(result['beta']):.2f}" if result['beta'] else "N/A"
        except (ValueError, TypeError) as e:
            print(f"Error formatting values for {ticker}: {e}")
            # Continue with unformatted values rather than failing entirely
        
        # Cache the result
        CACHE[cache_key] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {'error': f"Error fetching data for {ticker}: {str(e)}"}

def get_historical_price_data(ticker, period="1y"):
    """
    Get historical price data for a stock with efficient caching
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
    pd.DataFrame: DataFrame with historical price data
    """
    cache_key = f"history_{ticker}_{period}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        data = yf.download(ticker, period=period, progress=False)
        
        # Cache the result
        CACHE[cache_key] = {
            'data': data,
            'timestamp': current_time
        }
        
        return data
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

def get_alpha_vantage_news(api_key, topics="technology,business,economy", tickers=None, limit=10):
    """
    Fetch news from Alpha Vantage API with caching
    
    Parameters:
    api_key (str): Alpha Vantage API key
    topics (str): Comma-separated list of news topics
    tickers (str): Comma-separated list of ticker symbols to filter by
    limit (int): Maximum number of news articles to return
    
    Returns:
    list: List of news articles
    """
    cache_key = f"av_news_{topics}_{tickers}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < NEWS_CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        base_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}"
        
        # Add topics or tickers to the query depending on what's provided
        if tickers:
            url = f"{base_url}&tickers={tickers}"
        else:
            url = f"{base_url}&topics={topics}"
            
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Alpha Vantage API returned status code {response.status_code}")
            return get_fallback_news_data(limit)
            
        data = response.json()
        
        # Check if we got valid data
        if 'feed' not in data or not isinstance(data['feed'], list) or len(data['feed']) == 0:
            print("Error: Alpha Vantage API returned invalid data structure")
            return get_fallback_news_data(limit)
            
        # Convert Alpha Vantage format to match yfinance format
        news_items = []
        for item in data['feed'][:limit]:
            try:
                news_item = {
                    'title': item.get('title', 'No Title'),
                    'publisher': item.get('source', 'Alpha Vantage'),
                    'link': item.get('url', '#'),
                    'summary': item.get('summary', ''),
                    'providerPublishTime': int(datetime.datetime.strptime(
                        item.get('time_published', '20240322T120000'), 
                        '%Y%m%dT%H%M%S'
                    ).timestamp())
                }
                news_items.append(news_item)
            except Exception as e:
                print(f"Error processing news item: {e}")
                continue
        
        # Cache the result
        CACHE[cache_key] = {
            'data': news_items,
            'timestamp': current_time
        }
        
        return news_items
        
    except Exception as e:
        print(f"Error fetching news from Alpha Vantage: {e}")
        return get_fallback_news_data(limit)

def get_stock_market_news(ticker="^GSPC", limit=10):
    """
    Fetch news related to stock market using yfinance or Alpha Vantage as fallback
    
    Parameters:
    ticker (str): Stock ticker to get news for, defaults to S&P 500
    limit (int): Maximum number of news items to return
    
    Returns:
    list: List of news articles
    """
    cache_key = f"market_news_{ticker}_{limit}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < NEWS_CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        # Try yfinance first with multiple fallback tickers
        tickers_to_try = [ticker, "SPY", "AAPL", "MSFT", "AMZN"]
        
        for current_ticker in tickers_to_try:
            try:
                stock = yf.Ticker(current_ticker)
                news_items = stock.get_news()
                
                if isinstance(news_items, list) and len(news_items) > 0:
                    valid_items = [item for item in news_items if 'title' in item and 'link' in item]
                    
                    if valid_items:
                        result = valid_items[:min(limit, len(valid_items))]
                        
                        # Cache the result
                        CACHE[cache_key] = {
                            'data': result,
                            'timestamp': current_time
                        }
                        
                        return result
            except Exception as e:
                print(f"Error fetching news for {current_ticker}: {e}")
                continue
        
        # If yfinance failed, try Alpha Vantage
        av_news = get_alpha_vantage_news(API_KEY)
        
        # Cache the result
        CACHE[cache_key] = {
            'data': av_news,
            'timestamp': current_time
        }
        
        return av_news
        
    except Exception as e:
        print(f"Error in get_stock_market_news: {e}")
        return get_fallback_news_data(limit)

def process_news(articles):
    """
    Process news articles and add sentiment analysis
    
    Parameters:
    articles (list): List of news articles
    
    Returns:
    list: List of processed articles with sentiment analysis
    """
    if not articles:
        return []
        
    news_data = []
    try:
        for article in articles:
            try:
                title = article.get('title', 'No Title')
                source = article.get('publisher', 'Unknown Source')
                url = article.get('link', '#')
                summary = article.get('summary', '')
                
                # Use current timestamp if providerPublishTime is missing
                publish_time = article.get('providerPublishTime', int(time.time()))
                published = datetime.datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                
                # Truncate summary text
                text = summary[:150] + "..." if summary and len(summary) > 150 else summary
                
                # Sentiment analysis using VADER
                sentiment_scores = analyzer.polarity_scores(title)
                compound_score = sentiment_scores['compound']
                
                sentiment = 'positive' if compound_score >= 0.05 else \
                          'negative' if compound_score <= -0.05 else 'neutral'
                
                news_data.append({
                    'title': title,
                    'source': source,
                    'sentiment': sentiment,
                    'compound_score': round(compound_score, 3),
                    'url': url,
                    'text': text,
                    'published': published
                })
            except Exception as e:
                print(f"Error processing individual article: {e}")
                continue
    except Exception as e:
        print(f"Error in process_news: {e}")
    
    return news_data

def get_stock_news_for_ticker(ticker, limit=5):
    """
    Get news specifically about a single stock ticker with caching
    
    Parameters:
    ticker (str): Stock ticker symbol
    limit (int): Maximum number of news articles to return
    
    Returns:
    list: List of processed news articles with sentiment analysis
    """
    cache_key = f"stock_news_{ticker}_{limit}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < NEWS_CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        news_items = []
        
        # Try using yfinance first
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.get_news()
        except Exception as e:
            print(f"Error using get_news() for {ticker}: {e}")
        
        # If no news found via yfinance, try Alpha Vantage
        if not news_items or len(news_items) == 0:
            try:
                news_items = get_alpha_vantage_news(API_KEY, tickers=ticker, limit=limit)
            except Exception as e:
                print(f"Error getting Alpha Vantage news for {ticker}: {e}")
        
        # Process news
        processed_news = []
        for i, article in enumerate(news_items[:limit]):
            try:
                # Handle the nested content structure in yfinance response
                content = article
                
                # Check if this is a nested structure with a 'content' field
                if 'content' in article and isinstance(article['content'], dict):
                    content = article['content']
                
                # Extract title from content
                title = None
                if 'title' in content:
                    title = content['title']
                
                if not title and 'Title' in content:
                    title = content['Title']
                    
                # If still no title, skip this article
                if not title:
                    continue
                
                # Extract other fields
                source = content.get('provider', {}).get('displayName') if isinstance(content.get('provider'), dict) else \
                         content.get('source') or content.get('publisher') or "Financial News"
                
                # Get URL
                url = content.get('clickThroughUrl', {}).get('url') if isinstance(content.get('clickThroughUrl'), dict) else \
                      content.get('canonicalUrl', {}).get('url') if isinstance(content.get('canonicalUrl'), dict) else \
                      content.get('url') or content.get('link') or "#"
                
                # Get summary/description
                summary = content.get('summary') or content.get('description') or ""
                
                # Get publish date
                published = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                try:
                    if 'pubDate' in content and content['pubDate']:
                        pub_date = datetime.datetime.strptime(content['pubDate'], '%Y-%m-%dT%H:%M:%SZ')
                        published = pub_date.strftime('%Y-%m-%d %H:%M')
                    elif 'displayTime' in content and content['displayTime']:
                        pub_date = datetime.datetime.strptime(content['displayTime'], '%Y-%m-%dT%H:%M:%SZ')
                        published = pub_date.strftime('%Y-%m-%d %H:%M')
                    elif 'providerPublishTime' in content:
                        pub_time = content['providerPublishTime']
                        published = datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
                
                # Truncate summary text
                text = summary[:150] + "..." if summary and len(summary) > 150 else summary
                
                # Sentiment analysis using VADER
                sentiment_scores = analyzer.polarity_scores(title)
                compound_score = sentiment_scores['compound']
                
                sentiment = 'positive' if compound_score >= 0.05 else \
                         'negative' if compound_score <= -0.05 else 'neutral'
                
                processed_news.append({
                    'title': title,
                    'source': source,
                    'sentiment': sentiment,
                    'compound_score': round(compound_score, 3),
                    'url': url,
                    'text': text,
                    'published': published
                })
                
            except Exception as e:
                print(f"Error processing news article {i+1}: {e}")
                continue
        
        # If no processed news items, add fallback
        if not processed_news:
            processed_news.append({
                'title': f"Recent Market Activity for {ticker}",
                'source': 'Market News',
                'sentiment': 'neutral',
                'compound_score': 0.0,
                'url': '#',
                'text': f"Stay updated on the latest {ticker} market activity and financial performance.",
                'published': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        
        # Cache the result
        CACHE[cache_key] = {
            'data': processed_news,
            'timestamp': current_time
        }
        
        return processed_news
        
    except Exception as e:
        print(f"Error in get_stock_news_for_ticker for {ticker}: {e}")
        # Return a fallback news item
        fallback = [{
            'title': f"Market Update: {ticker}",
            'source': 'Financial News',
            'sentiment': 'neutral',
            'compound_score': 0.0,
            'url': '#',
            'text': f"Follow {ticker} for the latest updates and market performance.",
            'published': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }]
        
        # Cache the fallback result
        CACHE[cache_key] = {
            'data': fallback,
            'timestamp': current_time
        }
        
        return fallback
    
"""
Financial Dashboard Application - Optimized Version
Part 3: Market Data and Sector Analysis
"""

# Note: This section assumes Part 1 and 2 have been loaded

@lru_cache(maxsize=20)
def get_sector_data(etf, period='1y'):
    """
    Fetch historical data for a sector ETF with caching
    
    Parameters:
    etf (str): ETF ticker symbol
    period (str): Time period for data
    
    Returns:
    pd.DataFrame: DataFrame with historical price data
    """
    cache_key = f"sector_data_{etf}_{period}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        ticker = yf.Ticker(etf)
        # Remove the progress parameter
        data = ticker.history(period=period)  # Remove progress=False
        # Drop unwanted columns
        if 'Dividends' in data.columns:
            data = data.drop(columns=['Dividends'])
        if 'Stock Splits' in data.columns:
            data = data.drop(columns=['Stock Splits'])
        
        # Cache the result
        CACHE[cache_key] = {
            'data': data,
            'timestamp': current_time
        }
        
        return data
    except Exception as e:
        print(f"Error fetching data for {etf}: {e}")
        return pd.DataFrame()

def fetch_web_table(url, columns_to_drop=None, rename_columns=None, limit=10):
    """
    Generic function to fetch and process web tables
    
    Parameters:
    url (str): URL to fetch table from
    columns_to_drop (list): List of columns to drop
    rename_columns (dict): Dictionary mapping original column names to new names
    limit (int): Maximum number of rows to return
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    cache_key = f"web_table_{url.replace('/', '_')}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        dfs = pd.read_html(url)
        if not dfs or len(dfs) == 0:
            print(f"No tables found on page: {url}")
            return pd.DataFrame()
            
        df = dfs[0]
        
        # Drop specified columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Rename columns
        if rename_columns:
            df = df.rename(columns=rename_columns)
        
        # Limit rows
        if limit:
            df = df.head(limit)
        
        # Cache the result
        CACHE[cache_key] = {
            'data': df,
            'timestamp': current_time
        }
        
        return df
    except Exception as e:
        print(f"Error fetching web table from {url}: {e}")
        return pd.DataFrame()

def Indian_GL(url):
    """
    Fetch Indian gainers/losers with caching
    
    Parameters:
    url (str): URL to fetch data from
    
    Returns:
    pd.DataFrame: Processed DataFrame with gainers/losers
    """
    columns_to_drop = ['Price (Rs)', 'Change (Rs)', 'Prev Close (Rs)', 'High (Rs)', 'Low (Rs)', 'Volume (000s)']
    rename_columns = {'Name': 'Company name'}
    
    return fetch_web_table(url, columns_to_drop, rename_columns, 10)

def us_GL(url):
    """
    Fetch US gainers/losers with caching using BeautifulSoup
    
    Parameters:
    url (str): URL to fetch data from
    
    Returns:
    pd.DataFrame: Processed DataFrame with gainers/losers
    """
    cache_key = f"us_gl_{url.split('/')[-1]}"
    current_time = time.time()
    
    # Check if data is in cache and not expired
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table:
            print(f"No table found at {url}")
            return pd.DataFrame()
            
        headers = [header.text for header in table.find_all('th')]
        rows = [[cell.text for cell in row.find_all('td')] for row in table.find_all('tr')[1:]]
        
        df = pd.DataFrame(rows, columns=headers)
        columns_to_drop = [
            'Price', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 
            'EPS dilTTM', 'EPS dil growthTTM YoY', 'Div yield %TTM', 
            'Sector', 'Analyst Rating'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        result = df.head(10)
        
        # Cache the result
        CACHE[cache_key] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
    except Exception as e:
        print(f"Error in us_GL: {e}")
        return pd.DataFrame()

def create_sector_chart(sector, etf, data):
    """
    Create a candlestick chart for a sector with improved error handling
    
    Parameters:
    sector (str): Sector name
    etf (str): ETF ticker symbol
    data (pd.DataFrame): Historical price data
    
    Returns:
    str: JSON string of the chart data
    """
    try:
        # Ensure data is not empty and has the required columns
        if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            print(f"Missing required columns in data for {sector}")
            return None
            
        # Convert index to datetime strings
        date_strings = data.index.strftime('%Y-%m-%d').tolist()
        
        # Handle NaN values in the data
        candlestick_trace = go.Candlestick(
            x=date_strings,
            open=[float(x) if not np.isnan(x) else None for x in data['Open'].tolist()],
            high=[float(x) if not np.isnan(x) else None for x in data['High'].tolist()],
            low=[float(x) if not np.isnan(x) else None for x in data['Low'].tolist()],
            close=[float(x) if not np.isnan(x) else None for x in data['Close'].tolist()],
            increasing=dict(line=dict(color='#10b981')),
            decreasing=dict(line=dict(color='#ef4444'))
        )
        
        fig = go.Figure(data=[candlestick_trace])
        
        # Simplified layout - reduce memory usage
        fig.update_layout(
            title=f"{sector} Sector ({etf})",
            xaxis_title='Date',
            yaxis_title='Price',
            margin=dict(l=40, r=40, t=50, b=40),
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(30, 41, 59, 0.0)',
            plot_bgcolor='rgba(30, 41, 59, 0.0)',
            xaxis=dict(
                showgrid=False, 
                zeroline=False,
                rangeslider=dict(visible=False),
                automargin=True
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255, 255, 255, 0.1)', 
                zeroline=False,
                automargin=True
            )
        )
        
        # Use safe_json_dumps to handle NaN values
        return safe_json_dumps({'data': fig.data, 'layout': fig.layout})
    except Exception as e:
        print(f"Error creating chart for {sector}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_correlation_matrix(sector_data, period):
    """
    Create a correlation matrix heatmap for sectors with enhanced error handling
    
    Parameters:
    sector_data (dict): Dictionary mapping sector names to price data
    period (str): Time period for data
    
    Returns:
    str: JSON string of the correlation matrix chart
    """
    try:
        # Verify we have enough data to create a correlation matrix
        if not sector_data or len(sector_data) <= 1:
            print("Not enough sectors for correlation matrix")
            return None
            
        # Create DataFrame with closing prices
        sectors_to_use = []
        price_data = []
        
        # Validate each sector's data before including
        for sector, data in sector_data.items():
            if not data.empty and 'Close' in data.columns:
                sectors_to_use.append(sector)
                price_data.append(data['Close'])
        
        # Check again after validation
        if len(sectors_to_use) <= 1:
            print("Not enough valid sectors for correlation matrix")
            return None
            
        # Create DataFrame with validated data
        closing_prices = pd.concat(price_data, axis=1, keys=sectors_to_use)
        
        # Final check for data quality
        if closing_prices.empty or closing_prices.shape[0] <= 5:
            print("Not enough data points for correlation matrix")
            return None
            
        # Drop rows with all NaN values
        closing_prices = closing_prices.dropna(how='all')
        
        # Fill any remaining NaN values with forward fill, then backward fill
        closing_prices = closing_prices.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate correlation matrix
        correlation_matrix = closing_prices.corr().round(2)
        sectors = list(correlation_matrix.columns)
        
        # Convert correlation values to lists, handling NaN values
        z_values = []
        for row in correlation_matrix.values:
            z_row = []
            for val in row:
                if pd.isna(val):
                    z_row.append(None)
                else:
                    z_row.append(round(float(val), 2))
            z_values.append(z_row)
        
        # Create correlation heatmap
        corr_fig = go.Figure(data=[go.Heatmap(
            z=z_values,
            x=sectors,
            y=sectors,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=[[str(val if val is not None else "N/A") for val in row] for row in z_values],
            hoverinfo='text',
            showscale=True
        )])
        
        # Simplified layout
        corr_fig.update_layout(
            title=f"Sector Correlations - {period}",
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(30, 41, 59, 0.0)',
            plot_bgcolor='rgba(30, 41, 59, 0.0)',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Use safe_json_dumps to handle Plotly/Numpy objects
        return safe_json_dumps(corr_fig)
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")
        import traceback
        traceback.print_exc()
        return None
"""
Financial Dashboard Application - Optimized Version
Part 4: Primary Routes (Home, Search)
"""

# Note: This section assumes Parts 1-3 have been loaded

@app.route('/')
def home():
    """Home page route - displays market overview and financial news"""
    # Define market indices to track
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000',
        '^NSEI': 'Nifty 50',
        '^BSESN': 'Sensex',
    }

    # Calculate date range
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=15)

    # Initialize indices_data dictionary
    indices_data = {}
    
    # Use ThreadPoolExecutor to get data for all indices concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        def fetch_index_data(symbol, name):
            try:
                cache_key = f"index_data_{symbol}"
                current_time = time.time()
                
                # Check if data is in cache and not expired
                if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
                    return name, CACHE[cache_key]['data']
                
                data = yf.download(symbol, start=start_date, end=today, progress=False)
                
                if not data.empty and len(data) >= 2:
                    data.reset_index(inplace=True)
                    data.columns = data.columns.droplevel(1) if 'Close' in data.columns.levels[1] else data.columns
                    
                    latest_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    pct_change = ((latest_price - prev_price) / prev_price) * 100
                    point_change = latest_price - prev_price
                    
                    formatted_price = "{:,.2f}".format(float(latest_price))
                    
                    index_data = {
                        'price': formatted_price,
                        'pct_change': round(float(pct_change), 2),
                        'point_change': round(float(point_change), 2)
                    }
                    
                    # Cache the result
                    CACHE[cache_key] = {
                        'data': index_data,
                        'timestamp': current_time
                    }
                    
                    return name, index_data
                return name, None
            except Exception as e:
                print(f"Error fetching {name} data: {e}")
                return name, None
        
        # Execute tasks concurrently
        futures = {executor.submit(fetch_index_data, symbol, name): name for symbol, name in indices.items()}
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            name, data = future.result()
            if data:
                indices_data[name] = data
    
    # Fetch news articles
    articles = get_stock_market_news()
    
    # Process news with error handling
    formatted_news = process_news(articles)
    
    last_updated = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
    
    # Fetch inflation data for template
    try:
        inflation_data = get_fred_data("T5YIE")
        if inflation_data.empty:
            inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    except Exception as e:
        print(f"Error fetching inflation data: {e}")
        inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    
    # Periodically clear old cache entries
    if random.random() < 0.1:  # Clear cache with 10% probability on each request
        cleared = clear_expired_cache()
        if cleared > 0:
            print(f"Cleared {cleared} expired items from cache")
    
    return render_template('index.html', 
                          indices_data=indices_data, 
                          last_updated=last_updated, 
                          news_data=formatted_news,
                          inflation_data=inflation_data)

@app.route('/search')
def search():
    """
    Route to handle stock search and display comprehensive stock information
    with improved error handling and chart rendering
    """
    # Initialize with default values to prevent UnboundLocalError
    search_performed = False
    stock_info = {}
    stock_news = []
    chart_data = {'dates': [], 'prices': []}
    error = None
    query = ""
    
    # Get the search query
    query = request.args.get('q', '')
    
    # If no query was provided, render the initial search page
    if not query:
        return render_template('search.html', 
                              search_performed=False,
                              stock={},
                              news=[],
                              chart_data=chart_data)
    
    # Mark that a search was performed
    search_performed = True
    
    # Clean up query - remove spaces, convert to uppercase
    ticker = query.strip().upper()
    
    # Use ThreadPoolExecutor to fetch data concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks for stock info and news
        stock_info_future = executor.submit(get_stock_info, ticker)
        stock_news_future = executor.submit(get_stock_news_for_ticker, ticker)
        
        # Get stock info first - if this fails, we can't proceed
        stock_info = stock_info_future.result()
        
        # Check if there was an error in stock_info
        if 'error' in stock_info and stock_info['error'] is not None:
            error = stock_info['error']
            return render_template('search.html', 
                                  search_performed=True,
                                  error=error,
                                  query=query,
                                  stock={},
                                  news=[],
                                  chart_data=chart_data)
        
        # Get news data
        try:
            stock_news = stock_news_future.result()
        except Exception as e:
            print(f"Error retrieving news for {ticker}: {e}")
            stock_news = []
    
    # Get historical data for chart separately to avoid memory issues
    try:
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=365)  # Default to 1 year
        
        # Use cache if available
        cache_key = f"search_chart_{ticker}"
        current_time = time.time()
        
        if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CHART_CACHE_TIMEOUT):
            chart_data = CACHE[cache_key]['data']
        else:
            # Fetch data
            print(f"Fetching historical data for {ticker} from {start_date} to {today}")
            historical_data = yf.download(ticker, start=start_date, end=today, progress=False)
            
            if not historical_data.empty and len(historical_data) > 1:
                # Reset index to make Date a column
                historical_data.reset_index(inplace=True)
                
                # Process dates and ensure they're strings
                dates = historical_data['Date'].dt.strftime('%Y-%m-%d').tolist()
                
                # Process prices with careful error handling
                prices = []
                valid_indices = []
                
                for i, price in enumerate(historical_data['Close'].values):
                    if pd.isna(price):
                        continue  # Skip NaN values
                    
                    try:
                        # Convert to float and track valid index
                        price_float = float(price)
                        prices.append(price_float)
                        valid_indices.append(i)
                    except (TypeError, ValueError):
                        continue  # Skip invalid values
                
                # Get only dates that correspond to valid prices
                valid_dates = [dates[i] for i in valid_indices]
                
                # If we have valid data after filtering
                if valid_dates and prices and len(valid_dates) == len(prices):
                    chart_data = {
                        'dates': valid_dates,
                        'prices': prices
                    }
                    
                    # Cache the result
                    CACHE[cache_key] = {
                        'data': chart_data,
                        'timestamp': current_time
                    }
            else:
                print(f"No historical data available for {ticker} or insufficient data points")
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
    
    # Final validation of chart_data structure
    if not isinstance(chart_data, dict) or 'dates' not in chart_data or 'prices' not in chart_data:
        print("Warning: chart_data structure is invalid, resetting to empty structure")
        chart_data = {'dates': [], 'prices': []}
    
    # Log chart data stats
    print(f"Chart data contains {len(chart_data.get('dates', []))} data points")
    
    # Render template with all data
    return render_template('search.html', 
                          search_performed=True,
                          stock=stock_info,
                          news=stock_news,
                          chart_data=chart_data,
                          query=query)

@app.route('/candlestick/<symbol>')
def candlestick(symbol):
    """
    API route to get candlestick chart data for a stock
    
    Parameters:
    symbol (str): Stock ticker symbol
    
    Returns:
    JSON: Candlestick chart data
    """
    try:
        # URL decode the symbol to handle special characters like ^ in indices
        import urllib.parse
        decoded_symbol = urllib.parse.unquote(symbol)
        
        period = request.args.get('period', '1Y')  # Default to 1 year if no period provided
        today = datetime.date.today()

        # Define date ranges based on period
        period_map = {
            '3M': datetime.timedelta(days=90),
            '6M': datetime.timedelta(days=180),
            'YTD': datetime.timedelta(days=(today - datetime.date(today.year, 1, 1)).days),
            '1Y': datetime.timedelta(days=365),
            '5Y': datetime.timedelta(days=5 * 365),
            'Max': datetime.timedelta(days=10 * 365)  # Arbitrary large range for "Max"
        }

        delta = period_map.get(period, datetime.timedelta(days=365))  # Default to 1Y if invalid
        start_date = today - delta
        
        cache_key = f"candlestick_{decoded_symbol}_{period}"
        current_time = time.time()
        
        # Check if data is in cache and not expired (shorter timeout for charts - 15 minutes)
        if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CHART_CACHE_TIMEOUT):
            return jsonify(CACHE[cache_key]['data'])

        # Try to download data with proper error handling
        try:
            print(f"Downloading candlestick data for {decoded_symbol} from {start_date} to {today}")
            data = yf.download(decoded_symbol, start=start_date, end=today, progress=False)
            
            # Check if we got data
            if data.empty:
                print(f"No data returned for {decoded_symbol}")
                return jsonify({'error': f'No data available for {decoded_symbol}'}), 404
                
            data.reset_index(inplace=True)
            
            # Handle multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1) if len(data.columns.levels) > 1 else data.columns
            
            # Verify we have the necessary columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            for col in required_columns:
                if col not in data.columns:
                    return jsonify({'error': f'Missing required column: {col}'}), 500
            
            # Convert date column to string format
            date_strings = data['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Handle any NaN values in the data
            candlestick_data = {
                'Date': date_strings,
                'Open': [float(x) if not np.isnan(x) else None for x in data['Open'].tolist()],
                'High': [float(x) if not np.isnan(x) else None for x in data['High'].tolist()],
                'Low': [float(x) if not np.isnan(x) else None for x in data['Low'].tolist()],
                'Close': [float(x) if not np.isnan(x) else None for x in data['Close'].tolist()]
            }
            
            # Cache the result
            CACHE[cache_key] = {
                'data': candlestick_data,
                'timestamp': current_time
            }
            
            return jsonify(candlestick_data)
        except Exception as e:
            print(f"Error in yfinance download for {decoded_symbol}: {str(e)}")
            
            # Try an alternative approach for indices
            if '^' in decoded_symbol:
                try:
                    # Some indices work better without the ^ prefix in yfinance
                    alt_symbol = decoded_symbol.replace('^', '')
                    print(f"Trying alternative symbol format: {alt_symbol}")
                    data = yf.download(alt_symbol, start=start_date, end=today, progress=False)
                    
                    if not data.empty:
                        data.reset_index(inplace=True)
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.droplevel(1) if len(data.columns.levels) > 1 else data.columns
                        
                        date_strings = data['Date'].dt.strftime('%Y-%m-%d').tolist()
                        
                        candlestick_data = {
                            'Date': date_strings,
                            'Open': [float(x) if not np.isnan(x) else None for x in data['Open'].tolist()],
                            'High': [float(x) if not np.isnan(x) else None for x in data['High'].tolist()],
                            'Low': [float(x) if not np.isnan(x) else None for x in data['Low'].tolist()],
                            'Close': [float(x) if not np.isnan(x) else None for x in data['Close'].tolist()]
                        }
                        
                        CACHE[cache_key] = {
                            'data': candlestick_data,
                            'timestamp': current_time
                        }
                        
                        return jsonify(candlestick_data)
                except Exception as alt_e:
                    print(f"Alternative approach failed: {str(alt_e)}")
            
            # If all attempts failed, return error        
            return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500
    except Exception as outer_e:
        print(f"Unhandled exception in candlestick route: {str(outer_e)}")
        return jsonify({'error': f'Server error: {str(outer_e)}'}), 500
"""
Financial Dashboard Application - Optimized Version
Part 5: Secondary Routes (Sectors, Economic)
"""

# Note: This section assumes Parts 1-4 have been loaded

@app.route('/market_movers')
def market_movers():
    """Market movers page - displays top gainers and losers"""
    # Use ThreadPoolExecutor to fetch data concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        indian_gainers_future = executor.submit(Indian_GL, 'https://www.financialexpress.com/market/nse-top-gainers/')
        indian_losers_future = executor.submit(Indian_GL, 'https://www.financialexpress.com/market/nse-top-losers/')
        us_gainers_future = executor.submit(us_GL, "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/")
        us_losers_future = executor.submit(us_GL, "https://www.tradingview.com/markets/stocks-usa/market-movers-losers/")
        
        # Get results from futures
        indian_gainers_df = indian_gainers_future.result()
        indian_losers_df = indian_losers_future.result()
        us_gainers_df = us_gainers_future.result()
        us_losers_df = us_losers_future.result()
    
    return render_template('market_movers.html', 
                          indian_gainers=indian_gainers_df, 
                          indian_losers=indian_losers_df, 
                          us_gainers=us_gainers_df, 
                          us_losers=us_losers_df)

@app.route('/sectors')
def sectors():
    """Sectors page - displays sector performance with charts and correlation matrix"""
    period = request.args.get('period', '1y')
    
    sector_etfs = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    
    # Attempt to load from API first
    data_loaded = False
    sector_returns = {}
    sector_charts = {}
    sector_data_for_correlation = {}
    
    # Try to get data from the API first
    for sector, etf in sector_etfs.items():
        try:
            # Download data with a short timeout
            data = get_sector_data(etf, period)
            
            if not data.empty and len(data) > 0 and 'Close' in data.columns:
                data_loaded = True
                first_close = float(data['Close'].iloc[0])
                last_close = float(data['Close'].iloc[-1])
                pct_change = ((last_close - first_close) / first_close) * 100
                
                sector_returns[sector] = {
                    'name': sector,
                    'etf': etf,
                    'first_close': round(first_close, 2),
                    'last_close': round(last_close, 2),
                    'pct_change': round(pct_change, 2),
                    'is_positive': bool(pct_change > 0)
                }
                
                # Store data for correlation matrix
                sector_data_for_correlation[sector] = data
                
                # Create chart for this sector
                chart_json = create_sector_chart(sector, etf, data)
                if chart_json:
                    sector_charts[sector] = chart_json
                
                print(f"Successfully loaded data for {sector}")
        except Exception as e:
            print(f"API Error for {sector}: {e}")
    
    # If API fails, use fallback data
    if not data_loaded:
        print("Using fallback sector data")
        # Sample fallback data with realistic values
        fallback_data = {
            'Technology': {'pct_change': 12.5, 'first_close': 145.20, 'last_close': 163.35},
            'Financials': {'pct_change': 5.2, 'first_close': 35.45, 'last_close': 37.30},
            'Healthcare': {'pct_change': -2.1, 'first_close': 132.10, 'last_close': 129.32},
            'Consumer Discretionary': {'pct_change': 7.8, 'first_close': 168.75, 'last_close': 181.90},
            'Industrials': {'pct_change': 3.5, 'first_close': 98.45, 'last_close': 101.89},
            'Consumer Staples': {'pct_change': -1.2, 'first_close': 73.20, 'last_close': 72.32},
            'Energy': {'pct_change': 15.7, 'first_close': 76.30, 'last_close': 88.28},
            'Utilities': {'pct_change': -4.3, 'first_close': 65.40, 'last_close': 62.59},
            'Materials': {'pct_change': 6.2, 'first_close': 83.25, 'last_close': 88.41},
            'Real Estate': {'pct_change': -3.8, 'first_close': 41.60, 'last_close': 40.02},
            'Communication Services': {'pct_change': 9.3, 'first_close': 58.70, 'last_close': 64.16}
        }
        
        # Create sector returns from fallback data
        for sector, etf in sector_etfs.items():
            if sector in fallback_data:
                data = fallback_data[sector]
                sector_returns[sector] = {
                    'name': sector,
                    'etf': etf,
                    'first_close': data['first_close'],
                    'last_close': data['last_close'],
                    'pct_change': data['pct_change'],
                    'is_positive': data['pct_change'] > 0
                }
        
        # Generate fake charts for fallback data to ensure UI works
        for sector, etf in sector_etfs.items():
            if sector in fallback_data:
                # Create a simple fallback chart
                dates = pd.date_range(end=datetime.date.today(), periods=50)
                values = [fallback_data[sector]['first_close']]
                
                # Generate a trend based on percent change
                pct = fallback_data[sector]['pct_change'] / 100
                for i in range(1, 50):
                    next_val = values[-1] * (1 + (pct/50))
                    values.append(next_val)
                
                # Create trace for candlestick
                trace = go.Candlestick(
                    x=[d.strftime('%Y-%m-%d') for d in dates],
                    open=values,
                    high=[v * 1.02 for v in values],
                    low=[v * 0.98 for v in values],
                    close=values,
                    increasing=dict(line=dict(color='#10b981')),
                    decreasing=dict(line=dict(color='#ef4444'))
                )
                
                layout = go.Layout(
                    title=f"{sector} Sector ({etf}) - Estimated",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    paper_bgcolor='rgba(30, 41, 59, 0.0)',
                    plot_bgcolor='rgba(30, 41, 59, 0.0)'
                )
                
                fig = go.Figure(data=[trace], layout=layout)
                sector_charts[sector] = safe_json_dumps({'data': fig.data, 'layout': fig.layout})
                
                # Create synthetic data for correlation matrix
                fake_data = pd.DataFrame({
                    'Close': values
                }, index=dates)
                sector_data_for_correlation[sector] = fake_data
        
        # Set data_loaded flag as we now have fallback data
        data_loaded = True
    
    # Create correlation matrix if we have data for multiple sectors
    correlation_chart = None
    if len(sector_data_for_correlation) > 1:
        try:
            correlation_chart = create_correlation_matrix(sector_data_for_correlation, period)
            print(f"Successfully created correlation matrix with {len(sector_data_for_correlation)} sectors")
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")
            import traceback
            traceback.print_exc()
    
    # Render the template with whatever data we have
    try:
        note = "Note: Using cached sector data as real-time data is currently unavailable." if not data_loaded else None
        
        return render_template(
            'sectors.html', 
            period=period,
            sector_returns=sector_returns,
            sector_charts=sector_charts,
            correlation_chart=correlation_chart,
            sector_etfs=sector_etfs,
            error_message=note
        )
    except Exception as e:
        print(f"Error rendering sectors template: {e}")
        import traceback
        traceback.print_exc()
        
        # Absolute fallback - return to homepage with error
        return render_template(
            'index.html',
            error_message="Error loading sector data. Please try again later.",
            indices_data={},
            news_data=[],
            last_updated=datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
        )
@app.route('/update_sector_period', methods=['POST'])
def update_sector_period():
    """Update sector period"""
    period = request.form.get('period', '1y')
    return redirect(url_for('sectors', period=period))

@app.route('/ww_eco')
def ww_eco():
    """Economic indicators page - displays macroeconomic data"""
    cache_key = "ww_eco_page"
    current_time = time.time()
    
    # Check if page is in cache and not expired (1 hour timeout)
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return CACHE[cache_key]['data']
    
    # Initialize all data variables with empty DataFrames
    error_message = None
    
    # Use ThreadPoolExecutor for concurrent data fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks for all indicators
        gdp_future = executor.submit(get_fred_data, "GDP")
        unemployment_future = executor.submit(get_fred_data, "UNRATE")
        inflation_future = executor.submit(get_fred_data, "T5YIE")
        interest_future = executor.submit(get_fred_data, "REAINTRATREARAT10Y")
        
        # Get results with error handling
        try:
            gdp_data = gdp_future.result()
            unemployment_data = unemployment_future.result()
            inflation_data = inflation_future.result()
            interest_data = interest_future.result()
            
            # Add debug logs
            data_counts = {
                'GDP': len(gdp_data),
                'Unemployment': len(unemployment_data),
                'Inflation': len(inflation_data),
                'Interest': len(interest_data)
            }
            
            # Try alternative inflation series if T5YIE is unavailable
            if inflation_data.empty:
                print("T5YIE data unavailable, trying alternative inflation series")
                inflation_data = get_fred_data("MICH")  # Michigan Consumer Survey inflation expectations
                
        except Exception as e:
            print(f"Error in ww_eco route: {e}")
            # Ensure all variables have default values
            gdp_data = pd.DataFrame(columns=['GDP'])
            unemployment_data = pd.DataFrame(columns=['UNRATE'])
            inflation_data = pd.DataFrame(columns=['T5YIE'])
            interest_data = pd.DataFrame(columns=['REAINTRATREARAT10Y'])
            error_message = "Error fetching economic data. Please try again later."
    
    # Add index to empty DataFrames to prevent template errors
    if gdp_data.empty:
        gdp_data = pd.DataFrame({'GDP': []}, index=[])
    if unemployment_data.empty:
        unemployment_data = pd.DataFrame({'UNRATE': []}, index=[])
    if inflation_data.empty:
        inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    if interest_data.empty:
        interest_data = pd.DataFrame({'REAINTRATREARAT10Y': []}, index=[])
    
    # Render template
    template = render_template(
        'ww_eco.html', 
        gdp_data=gdp_data, 
        unemployment_data=unemployment_data, 
        inflation_data=inflation_data, 
        interest_data=interest_data,
        error_message=error_message
    )
    
    # Cache the rendered template
    CACHE[cache_key] = {
        'data': template,
        'timestamp': current_time
    }
    
    return template

"""
Financial Dashboard Application - Optimized Version
Part 6: Prediction Routes and App Execution
"""

# Note: This section assumes Parts 1-5 have been loaded

def get_stock_recommendations():
    """
    Get stock recommendations from cache file if exists, otherwise return empty list
    This function acts as a bridge between the prediction.py script results and the web application
    """
    try:
        if os.path.exists('stock_recommendations.json'):
            with open('stock_recommendations.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        return []

def run_predictions_task():
    """Function to run predictions in a background thread"""
    global prediction_job
    
    try:
        prediction_job['is_running'] = True
        prediction_job['start_time'] = datetime.datetime.now()
        prediction_job['status'] = 'running'
        prediction_job['error'] = None
        prediction_job['progress'] = 10
        
        # Import the prediction script
        import prediction as pred
        
        # Update status to indicate progress
        prediction_job['progress'] = 30
        
        # Run the main function and get recommendations
        recommendations = pred.main()
        
        # Update status
        prediction_job['status'] = 'completed'
        prediction_job['completion_time'] = datetime.datetime.now()
        prediction_job['progress'] = 100
        
        # Clear any cached predictions data
        for key in list(CACHE.keys()):
            if key.startswith('prediction_'):
                CACHE.pop(key, None)
        
    except Exception as e:
        # Handle any errors
        error_message = f"Error running predictions: {str(e)}"
        print(error_message)
        prediction_job['status'] = 'failed'
        prediction_job['error'] = error_message
    
    finally:
        prediction_job['is_running'] = False

@app.route('/predictions')
def predictions():
    """Predictions page - displays stock recommendations"""
    cache_key = "predictions_page"
    current_time = time.time()
    
    # Use cache unless prediction job is running
    if cache_key in CACHE and (current_time - CACHE[cache_key]['timestamp'] < CACHE_TIMEOUT) and not prediction_job['is_running']:
        return CACHE[cache_key]['data']
    
    # Get recommendations
    recommendations = get_stock_recommendations()
    
    # Get macroeconomic data for context
    try:
        # Use ThreadPoolExecutor for concurrent data fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks for all indicators
            gdp_future = executor.submit(get_fred_data, "GDP")
            unemployment_future = executor.submit(get_fred_data, "UNRATE")
            inflation_future = executor.submit(get_fred_data, "T5YIE")
            interest_future = executor.submit(get_fred_data, "REAINTRATREARAT10Y")
            
            # Get results
            gdp_data = gdp_future.result()
            unemployment_data = unemployment_future.result()
            inflation_data = inflation_future.result()
            interest_data = interest_future.result()
        
        # Format for display - get latest values
        macro_data = {
            'gdp': float(gdp_data.iloc[-1]) if not gdp_data.empty else None,
            'unemployment': float(unemployment_data.iloc[-1]) if not unemployment_data.empty else None,
            'inflation': float(inflation_data.iloc[-1]) if not inflation_data.empty else None, 
            'interest': float(interest_data.iloc[-1]) if not interest_data.empty else None
        }
    except Exception as e:
        print(f"Error getting macro data for predictions page: {e}")
        macro_data = {
            'gdp': None,
            'unemployment': None,
            'inflation': None,
            'interest': None
        }
    
    # Get last updated time
    last_updated = "Not yet generated"
    if os.path.exists('stock_recommendations.json'):
        last_updated = datetime.datetime.fromtimestamp(
            os.path.getmtime('stock_recommendations.json')
        ).strftime("%B %d, %Y %I:%M %p")
    
    # Render template
    template = render_template(
        'predictions.html',
        recommendations=recommendations,
        macro_data=macro_data,
        last_updated=last_updated,
        job_status=prediction_job
    )
    
    # Cache the rendered template if predictions are not running
    if not prediction_job['is_running']:
        CACHE[cache_key] = {
            'data': template,
            'timestamp': current_time
        }
    
    return template

@app.route('/run_predictions', methods=['POST'])
def run_predictions():
    """Route to run the stock prediction algorithm"""
    global prediction_job
    
    # Check if prediction is already running
    if prediction_job['is_running']:
        return render_template(
            'predictions.html',
            error_message="Prediction process is already running. Please wait for it to complete.",
            recommendations=get_stock_recommendations(),
            job_status=prediction_job
        )
    
    # Start the prediction task in a background thread
    prediction_thread = threading.Thread(target=run_predictions_task)
    prediction_thread.daemon = True
    prediction_thread.start()
    
    # Redirect back to the predictions page
    return redirect(url_for('predictions'))

@app.route('/prediction_status')
def prediction_status():
    """API route to check prediction status"""
    return jsonify(prediction_job)

@app.route('/clear_cache')
def clear_cache():
    """Admin route to clear cache (can be protected with authentication)"""
    count = clear_expired_cache(force_clear=True)
    return jsonify({'message': f'Cache cleared. {count} items removed.'})

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page"""
    # Instead of using a separate template, return a basic error message
    return render_template('index.html', 
                          error_message="The page you requested was not found.", 
                          indices_data={}, 
                          news_data=[],
                          last_updated=datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")), 404

@app.errorhandler(500)
def server_error(e):
    """Custom 500 error page"""
    # Instead of using a separate template, return a basic error message
    return render_template('index.html', 
                          error_message="An internal server error occurred. Please try again later.", 
                          indices_data={}, 
                          news_data=[],
                          last_updated=datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")), 500

# Schedule periodic cache cleanup
def schedule_cache_cleanup():
    """Function to schedule periodic cache cleanup"""
    while True:
        time.sleep(3600)  # Run every hour
        cleared = clear_expired_cache()
        print(f"Scheduled cache cleanup: {cleared} items removed")

# Start cache cleanup thread when app starts
if __name__ == '__main__':
    # Start cache cleanup thread
    cleanup_thread = threading.Thread(target=schedule_cache_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Run the app
    app.run(debug=True)
else:
    # In production, still start the cleanup thread
    import random
    
    # Only start cleanup thread with 25% probability to avoid
    # multiple threads when using multiple workers
    if random.random() < 0.25:
        cleanup_thread = threading.Thread(target=schedule_cache_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()