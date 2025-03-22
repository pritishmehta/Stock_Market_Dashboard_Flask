from flask import Flask, render_template, jsonify, request
import yfinance as yf
import datetime 
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import numpy as np
import plotly.graph_objects as go
import plotly
import json
from flask import redirect, url_for
import pandas_datareader as pdr
app = Flask(__name__)
import time

# Replace with your actual Alpha Vantage API key
API_KEY = '5DHU06PG79LY13BA'

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Improved function to fetch news from Alpha Vantage with better error handling
analyzer = SentimentIntensityAnalyzer()  # Initialize VADER sentiment analyzer

def get_stock_info(ticker):
    """
    Get comprehensive information about a stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Stock information including price, 52 week high/low, etc.
    """
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
        
        return result
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {'error': f"Error fetching data for {ticker}: {str(e)}"}

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

def get_stock_news_for_ticker(ticker, limit=5):
    """
    Get news specifically about a single stock ticker with improved error handling
    
    Parameters:
    ticker (str): Stock ticker symbol
    limit (int): Maximum number of news items to return
    
    Returns:
    list: List of processed news articles with sentiment analysis
    """
    try:
        print(f"Fetching news for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        news_items = []
        
        try:
            # Try to get news using get_news method
            news_items = stock.get_news()
            print(f"Retrieved {len(news_items) if news_items else 0} news items via yfinance get_news()")
        except Exception as e:
            print(f"Error using get_news() for {ticker}: {e}")
            news_items = []
        
        # If no news found via yfinance, try Alpha Vantage
        if not news_items or len(news_items) == 0:
            print(f"No news found via yfinance for {ticker}, trying Alpha Vantage...")
            # Try Alpha Vantage as fallback
            try:
                tickers_param = ticker
                news_items = get_alpha_vantage_news(API_KEY, tickers=tickers_param, limit=limit)
                print(f"Retrieved {len(news_items) if news_items else 0} news items via Alpha Vantage")
            except Exception as e:
                print(f"Error getting Alpha Vantage news for {ticker}: {e}")
                news_items = []
        
        # If still no news, use fallback data
        if not news_items or len(news_items) == 0:
            print(f"No news found for {ticker}, using fallback data...")
            news_items = get_fallback_news_data(limit)
            # Customize fallback data for this ticker
            for item in news_items:
                item['title'] = item['title'].replace('Markets', f"{ticker} Stock")
                
        # Process news with proper error handling
        processed_news = []
        for article in news_items[:limit]:
            try:
                # Extract basic article information with fallbacks
                title = article.get('title', 'No Title')
                if not title or title == 'No Title':
                    # Skip articles without titles
                    continue
                    
                source = article.get('publisher', article.get('source', 'Financial News'))
                url = article.get('link', article.get('url', '#'))
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
                print(f"Error processing news article: {e}")
                continue
        
        # Log detailed information about the processed news
        print(f"Successfully processed {len(processed_news)} news items for {ticker}")
        if processed_news:
            print(f"First news item: {processed_news[0]['title']}")
        else:
            print("No news was successfully processed")
            
            # If processing failed for all items, add at least one fallback item
            processed_news.append({
                'title': f"Recent Market Activity for {ticker}",
                'source': 'Market News',
                'sentiment': 'neutral',
                'compound_score': 0.0,
                'url': '#',
                'text': f"Stay updated on the latest {ticker} market activity and financial performance.",
                'published': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            
        return processed_news
        
    except Exception as e:
        print(f"Error in get_stock_news_for_ticker for {ticker}: {e}")
        # Return a fallback news item to ensure the news section isn't empty
        return [{
            'title': f"Market Update: {ticker}",
            'source': 'Financial News',
            'sentiment': 'neutral',
            'compound_score': 0.0,
            'url': '#',
            'text': f"Follow {ticker} for the latest updates and market performance.",
            'published': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        }]

# Add this route to handle stock search
@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    if not query:
        return render_template('search.html', 
                              search_performed=False,
                              stock={},
                              news=[],
                              chart_data={'dates': [], 'prices': []})
    
    # Clean up query - remove spaces, convert to uppercase
    ticker = query.strip().upper()
    print(f"Processing search for ticker: {ticker}")
    
    # Get stock information
    try:
        stock_info = get_stock_info(ticker)
        
        # Check if there was an error
        if 'error' in stock_info and stock_info['error'] is not None:
            print(f"Error in stock data: {stock_info['error']}")
            return render_template('search.html', 
                                  search_performed=True,
                                  error=stock_info['error'],
                                  query=query,
                                  stock={},
                                  news=[],
                                  chart_data={'dates': [], 'prices': []})
    except Exception as e:
        print(f"Exception in get_stock_info for {ticker}: {str(e)}")
        return render_template('search.html', 
                              search_performed=True,
                              error=f"Error processing stock data: {str(e)}",
                              query=query,
                              stock={},
                              news=[],
                              chart_data={'dates': [], 'prices': []})
    
    # Get news for this stock with improved error handling
    stock_news = []
    try:
        stock_news = get_stock_news_for_ticker(ticker)
        print(f"Retrieved {len(stock_news)} news items for {ticker}")
        
        # Debug the first news item to verify structure
        if stock_news and len(stock_news) > 0:
            print(f"First news item title: {stock_news[0]['title']}")
            print(f"News item keys: {stock_news[0].keys()}")
        else:
            print("No news items were returned")
            # Ensure we at least have an empty list
            stock_news = []
    except Exception as e:
        print(f"Error retrieving news for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        # Set default empty news list
        stock_news = []
    
    # Get historical data for chart 
    chart_data = {'dates': [], 'prices': []}
    try:
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=365)  # Default to 1 year
        
        print(f"Fetching historical data for {ticker} from {start_date} to {today}")
        historical_data = yf.download(ticker, start=start_date, end=today)
        
        if not historical_data.empty and len(historical_data) > 1:
            historical_data.reset_index(inplace=True)
            
            # Convert to a list of strings for dates
            dates = historical_data['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Fix the prices conversion with better error handling
            prices = []
            for price in historical_data['Close'].values:
                # Handle different data types that might be returned
                if isinstance(price, (int, float)):
                    # If it's already a number, just append it
                    prices.append(float(price))
                elif isinstance(price, list):
                    # If it's a list, take the first numeric value if available
                    print(f"WARNING: Found a list where a single value was expected: {price}")
                    for item in price:
                        if isinstance(item, (int, float)):
                            prices.append(float(item))
                            break
                    else:
                        # If no numeric value found, use 0
                        prices.append(0.0)
                elif price is None or pd.isna(price):
                    # Handle None or NaN values
                    prices.append(None)
                else:
                    # Try converting string or other types
                    try:
                        prices.append(float(price))
                    except (TypeError, ValueError):
                        # If conversion fails, use None
                        print(f"WARNING: Could not convert price value to float: {price}")
                        prices.append(None)
            
            # Remove any None values from prices and corresponding dates
            valid_data = [(d, p) for d, p in zip(dates, prices) if p is not None]
            if valid_data:
                dates = [item[0] for item in valid_data]
                prices = [item[1] for item in valid_data]
                
                # Debug output
                print(f"Processed {len(prices)} valid price points out of {len(historical_data)} records")
            else:
                print("No valid price data found after filtering")
            
            # Prepare data for the chart
            chart_data = {
                'dates': dates,
                'prices': prices
            }
        else:
            print(f"No historical data available for {ticker} or insufficient data points")
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Rendering template with {len(stock_news)} news items and {len(chart_data['dates'])} chart points")
    
    # Render template with all data
    return render_template('search.html', 
                          search_performed=True,
                          stock=stock_info,
                          news=stock_news,
                          chart_data=chart_data,
                          query=query)

def get_stock_market_news(ticker="^GSPC", limit=10, max_retries=3, retry_delay=1):
    """
    Fetch news related to stock market using yfinance with improved reliability
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count+1} to fetch news for {ticker}")
            
            # Try multiple tickers if one fails
            tickers_to_try = [ticker, "SPY", "AAPL", "MSFT", "AMZN"]
            
            for current_ticker in tickers_to_try:
                try:
                    print(f"Trying to fetch news for {current_ticker}")
                    stock = yf.Ticker(current_ticker)
                    
                    # Important: Use stock.get_news() instead of stock.news for more consistent results
                    # The .news property can sometimes be inconsistent
                    news_items = stock.get_news()
                    
                    if isinstance(news_items, list) and len(news_items) > 0:
                        print(f"Successfully fetched {len(news_items)} news items from {current_ticker}")
                        
                        # Validate the news item structure
                        valid_items = []
                        for item in news_items:
                            if 'title' in item and 'link' in item:
                                valid_items.append(item)
                        
                        if valid_items:
                            return valid_items[:min(limit, len(valid_items))]
                except Exception as e:
                    print(f"Error fetching news for {current_ticker}: {e}")
                    continue  # Try the next ticker
            
            # If we get here, none of the tickers worked
            print("None of the tickers provided valid news")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"Error in news fetch attempt: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
    
    print("All retry attempts failed, using fallback news data")
    return get_fallback_news_data(limit)

def get_alpha_vantage_news(api_key, topics="technology,business,economy", tickers=None, limit=10):
    """
    Fetch news from Alpha Vantage API instead of yfinance
    
    Parameters:
    api_key (str): Alpha Vantage API key
    topics (str): Comma-separated list of news topics
    tickers (str): Comma-separated list of ticker symbols to filter by
    limit (int): Maximum number of news articles to return
    
    Returns:
    list: List of news articles in a format similar to yfinance
    """
    try:
        base_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}"
        
        # Add topics or tickers to the query depending on what's provided
        if tickers:
            url = f"{base_url}&tickers={tickers}"
            print(f"Fetching news from Alpha Vantage API for tickers: {tickers}")
        else:
            url = f"{base_url}&topics={topics}"
            print(f"Fetching news from Alpha Vantage API with topics: {topics}")
            
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Alpha Vantage API returned status code {response.status_code}")
            return get_fallback_news_data(limit)
            
        data = response.json()
        
        # Check if we got valid data
        if 'feed' not in data or not isinstance(data['feed'], list) or len(data['feed']) == 0:
            print("Error: Alpha Vantage API returned invalid data structure")
            print(f"Response keys: {data.keys()}")
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
                
        print(f"Successfully fetched {len(news_items)} news items from Alpha Vantage")
        return news_items
        
    except Exception as e:
        print(f"Error fetching news from Alpha Vantage: {e}")
        return get_fallback_news_data(limit)

def get_fallback_news_data(limit=5):
    """Generate fallback news data when API fails"""
    fallback_news = [
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
    ]
    return fallback_news[:limit]

def process_news(articles):
    """Process news articles and add sentiment analysis"""
    if not articles:
        print("No articles to process")
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
                published = datetime.datetime.fromtimestamp(publish_time).strftime('%Y%m%dT%H%M%S')
                
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
                # Continue with next article instead of failing entirely
                continue
    except Exception as e:
        print(f"Error in process_news: {e}")
    
    print(f"Processed {len(news_data)} articles")
    return news_data

@app.route('/')
def home():
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000',
        '^NSEI': 'Nifty 50',
        '^BSESN': 'Sensex',
    }

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=15)

    indices_data = {}
    for symbol, name in indices.items():
        try:
            data = yf.download(symbol, start=start_date, end=today)
            data.reset_index(inplace=True)
            data.columns = data.columns.droplevel(1) if 'Close' in data.columns.levels[1] else data.columns
            
            if not data.empty and len(data) >= 2:
                latest_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                pct_change = ((latest_price - prev_price) / prev_price) * 100
                point_change = latest_price - prev_price
                
                formatted_price = "{:,.2f}".format(float(latest_price))
                
                indices_data[name] = {
                    'price': formatted_price,
                    'pct_change': round(float(pct_change), 2),
                    'point_change': round(float(point_change), 2)
                }
        except Exception as e:
            print(f"Error fetching {name} data: {e}")
    
    # Try multiple methods to get news - first try yfinance
    print("Attempting to fetch news using yfinance...")
    articles = get_stock_market_news()
    
    # If yfinance failed or returned no valid results, try Alpha Vantage
    if not articles or len(articles) == 0 or articles[0].get('title') == 'Markets React to Recent Economic Data':
        print("yfinance news retrieval failed or returned fallback data, trying Alpha Vantage...")
        articles = get_alpha_vantage_news(API_KEY)
    
    print(f"Final news article count: {len(articles)}")
    
    # Process news with error handling
    formatted_news = process_news(articles)
    
    # Debug - print first news item
    if formatted_news and len(formatted_news) > 0:
        print("First news item after processing:")
        print(formatted_news[0])
    else:
        print("No news items after processing!")
    
    last_updated = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
    
    # Add inflation_data to fix the template error
    try:
        inflation_data = get_fred_data("T5YIE")
        if inflation_data.empty:
            inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    except Exception as e:
        print(f"Error fetching inflation data: {e}")
        inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    
    return render_template('index.html', 
                          indices_data=indices_data, 
                          last_updated=last_updated, 
                          news_data=formatted_news,
                          inflation_data=inflation_data)


def Indian_GL(url):
    try:
        dfs = pd.read_html(url)
        if not dfs or len(dfs) == 0:
            print(f"No tables found on page: {url}")
            return pd.DataFrame()  # Return empty DataFrame instead of None
        df = dfs[0]
        df = df.drop(columns=['Price (Rs)', 'Change (Rs)', 'Prev Close (Rs)', 'High (Rs)', 'Low (Rs)', 'Volume (000s)'], errors='ignore')
        df = df.head(10)
        df.rename(columns={'Name': 'Company name'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error in Indian_GL: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def us_GL(url):
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
        df = df.drop(columns=['Price', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 
                             'EPS dilTTM', 'EPS dil growthTTM YoY', 'Div yield %TTM', 'Sector', 'Analyst Rating'], 
                     errors='ignore')
        return df.head(10)
    except Exception as e:
        print(f"Error in us_GL: {e}")
        return pd.DataFrame()
# Get top gainers

@app.route('/market_movers')
def market_movers():
    indian_gainers_df = Indian_GL('https://www.financialexpress.com/market/nse-top-gainers/')
    indian_losers_df = Indian_GL('https://www.financialexpress.com/market/nse-top-losers/')
    US_gainers_df = us_GL("https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/")
    US_losers_df = us_GL("https://www.tradingview.com/markets/stocks-usa/market-movers-losers/")
    
    return render_template('market_movers.html', 
                          indian_gainers=indian_gainers_df, 
                          indian_losers=indian_losers_df, 
                          us_gainers=US_gainers_df, 
                          us_losers=US_losers_df)

@app.route('/candlestick/<symbol>')
def candlestick(symbol):
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

    try:
        data = yf.download(symbol, start=start_date, end=today)
        data.reset_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        if not data.empty:
            candlestick_data = {
                'Date': data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'Open': data['Open'].tolist(),
                'High': data['High'].tolist(),
                'Low': data['Low'].tolist(),
                'Close': data['Close'].tolist()
            }
            return jsonify(candlestick_data)
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Function to fetch historical data for a sector ETF
def get_sector_data(etf, period='1y'):
    """
    Fetch historical data for a sector ETF
    """
    try:
        ticker = yf.Ticker(etf)
        data = ticker.history(period=period)
        # Drop unwanted columns
        if 'Dividends' in data.columns:
            data = data.drop(columns=['Dividends'])
        if 'Stock Splits' in data.columns:
            data = data.drop(columns=['Stock Splits'])
        return data
    except Exception as e:
        print(f"Error fetching data for {etf}: {e}")
        return pd.DataFrame()

# Add this route to your app.py file
# Complete fixed sectors route
@app.route('/sectors')
def sectors():
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
    
    sector_data = {}
    sector_returns = {}
    sector_charts = {}
    
    # Track if any data was successfully loaded
    data_loaded = False
    
    for sector, etf in sector_etfs.items():
        data = get_sector_data(etf, period)
        if not data.empty and len(data) > 0:
            data_loaded = True
            sector_data[sector] = data
            first_close = data['Close'].iloc[0]
            last_close = data['Close'].iloc[-1]
            pct_change = ((last_close - first_close) / first_close) * 100
            sector_returns[sector] = {
                'name': sector,
                'etf': etf,
                'first_close': round(first_close, 2),
                'last_close': round(last_close, 2),
                'pct_change': round(pct_change, 2),
                'is_positive': pct_change > 0
            }
            # Inside the sectors() function, update the chart creation logic
            try:
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index.strftime('%Y-%m-%d').tolist(),
                    open=data['Open'].tolist(),
                    high=data['High'].tolist(),
                    low=data['Low'].tolist(),
                    close=data['Close'].tolist(),
                    increasing=dict(line=dict(color='#10b981')),
                    decreasing=dict(line=dict(color='#ef4444'))
                )])
                fig.update_layout(
                    title=f"{sector} Sector ({etf})",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=500,  # Increased height for modal view
                    template="plotly_dark",
                    paper_bgcolor='rgba(30, 41, 59, 0.0)',
                    plot_bgcolor='rgba(30, 41, 59, 0.0)',
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False,
                        rangeslider=dict(visible=False),  # Hide range slider
                        automargin=True
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255, 255, 255, 0.1)', 
                        zeroline=False,
                        automargin=True
                    ),
                    dragmode='zoom',
                    modebar=dict(
                        orientation='v',
                        bgcolor='rgba(30, 41, 59, 0.7)'
                    ),
                    hoverlabel=dict(
                        bgcolor='rgba(30, 41, 59, 0.8)',
                        font=dict(color='white')
                    )
                )
                # Add more interactive options to the config
                config = {
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
                }
                chart_json = json.dumps({'data': fig.data, 'layout': fig.layout}, cls=plotly.utils.PlotlyJSONEncoder)
                sector_charts[sector] = chart_json
            except Exception as e:
                print(f"Error creating chart for {sector}: {e}")    
    
    # Handle the case where no data was loaded (all API calls failed)
    if not data_loaded:
        # Provide a default empty structure for the template
        print("No sector data was loaded. Providing default empty structure.")
        # Return a message to the template
        return render_template(
            'sectors.html',
            period=period,
            sector_returns={},
            sector_charts={},
            correlation_chart=None,
            sector_etfs=sector_etfs,
            error_message="Unable to load sector data. Please try again later."
        )
    
    # Correlation matrix logic
    correlation_chart = None
    try:
        if sector_data:
            print(f"Creating correlation matrix with {len(sector_data)} sectors")
            closing_prices = pd.DataFrame({sector: data['Close'] for sector, data in sector_data.items()})
            print(f"Closing prices shape: {closing_prices.shape}")
            if not closing_prices.empty and closing_prices.shape[0] > 5:
                correlation_matrix = closing_prices.corr().round(2)
                print(f"Correlation matrix shape: {correlation_matrix.shape}")
                sectors = list(correlation_matrix.columns)
                z_values = correlation_matrix.values.tolist()
                corr_fig = go.Figure(data=[go.Heatmap(
                    z=z_values,
                    x=sectors,
                    y=sectors,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=[[str(round(val, 2)) for val in row] for row in z_values],
                    hoverinfo='text',
                    showscale=True
                )])
                corr_fig.update_layout(
                    title=f"Sector Correlations - {period}",
                    height=600,
                    template="plotly_dark",
                    paper_bgcolor='rgba(30, 41, 59, 0.0)',
                    plot_bgcolor='rgba(30, 41, 59, 0.0)',
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                correlation_chart = json.dumps(corr_fig, cls=plotly.utils.PlotlyJSONEncoder)
                print(f"Correlation chart JSON length: {len(correlation_chart)}")
            else:
                print("Not enough data for correlation matrix")
        else:
            print("No sector data for correlation matrix")
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")
    
    return render_template(
        'sectors.html', 
        period=period,
        sector_returns=sector_returns,
        sector_charts=sector_charts,
        correlation_chart=correlation_chart,
        sector_etfs=sector_etfs,
        error_message=None
    )

# Add this route to your app.py
@app.route('/update_sector_period', methods=['POST'])
def update_sector_period():
    period = request.form.get('period', '1y')
    return redirect(url_for('sectors', period=period))

# Define helper function outside the route# Replace your existing get_fred_data function with this improved version

def get_fred_data(series_id):
    """
    Fetch data from FRED with proper error handling and formatting.
    Always returns a DataFrame with the appropriate column, even if empty.
    """
    try:
        # Attempt to fetch data from FRED
        data = pdr.DataReader(series_id, 'fred')
        
        # Check if data was successfully retrieved
        if data.empty:
            print(f"Warning: Empty DataFrame returned for {series_id}")
            # Return empty DataFrame with correct column
            return pd.DataFrame(columns=[series_id])
            
        # Return the data
        return data
        
    except Exception as e:
        print(f"Error fetching FRED data for {series_id}: {e}")
        # Return empty DataFrame with proper column name
        return pd.DataFrame(columns=[series_id])
    
# Update your ww_eco route to include better error handling
@app.route('/ww_eco')
def ww_eco():
    # Initialize all data variables with empty DataFrames
    gdp_data = pd.DataFrame()
    unemployment_data = pd.DataFrame()
    inflation_data = pd.DataFrame()
    interest_data = pd.DataFrame()
    error_message = None
    
    try:
        # Attempt to fetch the data
        gdp_data = get_fred_data("GDP")
        unemployment_data = get_fred_data("UNRATE")
        inflation_data = get_fred_data("T5YIE")
        interest_data = get_fred_data("REAINTRATREARAT10Y")
        
        # Add logging to help debug
        print(f"GDP data: {len(gdp_data)} rows")
        print(f"Unemployment data: {len(unemployment_data)} rows")
        print(f"Inflation data: {len(inflation_data)} rows")
        print(f"Interest data: {len(interest_data)} rows")
        
        if inflation_data.empty:
            # Try an alternative inflation series if T5YIE is unavailable
            print("T5YIE data unavailable, trying alternative inflation series")
            inflation_data = get_fred_data("MICH")  # Michigan Consumer Survey inflation expectations
            if inflation_data.empty:
                error_message = "Inflation data currently unavailable."
                # Ensure inflation_data has the right column even if empty
                inflation_data = pd.DataFrame(columns=['T5YIE'])
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
    print(f"DEBUG - Inflation data: {inflation_data.head()}")
    
    return render_template('ww_eco.html', 
                          gdp_data=gdp_data, 
                          unemployment_data=unemployment_data, 
                          inflation_data=inflation_data, 
                          interest_data=interest_data,
                          error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
