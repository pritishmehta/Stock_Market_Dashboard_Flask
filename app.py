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

def get_stock_market_news(ticker="^GSPC", limit=10, max_retries=3, retry_delay=1):
    """
    Fetch news related to stock market using yfinance with retry mechanism
    
    Parameters:
    ticker (str): Stock ticker symbol (default: ^GSPC for S&P 500)
    limit (int): Maximum number of news articles to return
    max_retries (int): Maximum number of retry attempts
    retry_delay (int): Delay between retries in seconds
    
    Returns:
    list: List of news articles or fallback data if failed
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Create yfinance Ticker object
            market = yf.Ticker(ticker)
            
            # Get news items
            news_items = market.news
            
            if news_items:
                # Return limited number of articles
                return news_items[:min(limit, len(news_items))]
            else:
                print(f"No news items returned for ticker {ticker}")
                
            # Increment retry count and wait before next attempt
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"Error fetching news from yfinance: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
    
    # Return fallback data if all retries failed
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
    news_data = []
    for article in articles:
        title = article.get('title', 'No Title')
        source = article.get('publisher', 'Unknown Source')
        url = article.get('link', '#')
        summary = article.get('summary', '')
        published = datetime.datetime.fromtimestamp(
            article.get('providerPublishTime', 0)
        ).strftime('%Y%m%dT%H%M%S')
        
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
                
                # Fix: Convert to float first, then format
                formatted_price = "{:,.2f}".format(float(latest_price))
                
                indices_data[name] = {
                    'price': formatted_price,
                    'pct_change': round(float(pct_change), 2),  # Ensure it's a float
                    'point_change': round(float(point_change), 2)  # Ensure it's a float
                }
        except Exception as e:
            print(f"Error fetching {name} data: {e}")
    
    # Get news articles from Alpha Vantage
    articles = get_stock_market_news()

    print(f"Retrieved {len(articles)} articles from Alpha Vantage")
    formatted_news = process_news(articles)
    last_updated = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
    
    # Add inflation_data to fix the template error
    try:
        # Try to get inflation data from FRED - similar to your ww_eco route
        inflation_data = get_fred_data("T5YIE")
        if inflation_data.empty:
            # Create a minimal placeholder DataFrame with proper structure
            inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    except Exception as e:
        print(f"Error fetching inflation data: {e}")
        # Create a minimal placeholder DataFrame with proper structure
        inflation_data = pd.DataFrame({'T5YIE': []}, index=[])
    
    return render_template('index.html', 
                          indices_data=indices_data, 
                          last_updated=last_updated, 
                          news_data=formatted_news,
                          inflation_data=inflation_data)  # Add this parameter


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
