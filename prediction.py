import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import fredapi as fred
import warnings
warnings.filterwarnings('ignore')

# API keys
API_KEYS = {
    'alpha_vantage': 'UANQ1XFO30S3JSDD',
    'finnhub': 'cur3nb1r01qifa4t9dogcur3nb1r01qifa4t9dp0',
    'fmp': 'Kt1O7bBlrbcTGG4tNPpN2kBRSb3W8XDw',
    'marketaux': 'Jx0gR43HirosrcniOFopsB1urbeyr7Ik4zi47WLW',
    'news_api': '043a445e570c4965a37bd08abe169a7d',
    'quandl': 'fdKTrrAEbzjpf_Ez7q1h',
    'fred': 'a4a9c77dac746ee7942df32a68a0bccf'
}

# FRED API client
fred_api_key = 'a4a9c77dac746ee7942df32a68a0bccf'
fred_client = fred.Fred(api_key=fred_api_key)

# Define the S&P 500 companies list (or any other universe of stocks you want to analyze)
def get_sp500_companies():
    """Get list of S&P 500 companies using Wikipedia"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()
    except:
        # Fallback to a sample of major companies if the above fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 
                'PG', 'UNH', 'HD', 'BAC', 'MA', 'XOM', 'DIS', 'ADBE', 'CRM', 'NFLX',
                'CSCO', 'INTC', 'PFE', 'T', 'VZ', 'KO', 'PEP', 'WMT', 'ABT', 'MRK']

# 1. Financial Performance Data Collection
def get_financial_data(ticker):
    """Get financial data from Financial Modeling Prep"""
    base_url = "https://financialmodelingprep.com/api/v3"
    
    # Get income statement
    income_url = f"{base_url}/income-statement/{ticker}?apikey={API_KEYS['fmp']}&limit=4"
    income_response = requests.get(income_url)
    
    # Get balance sheet
    balance_url = f"{base_url}/balance-sheet-statement/{ticker}?apikey={API_KEYS['fmp']}&limit=4"
    balance_response = requests.get(balance_url)
    
    # Get cash flow
    cashflow_url = f"{base_url}/cash-flow-statement/{ticker}?apikey={API_KEYS['fmp']}&limit=4"
    cashflow_response = requests.get(cashflow_url)
    
    # Get key metrics
    metrics_url = f"{base_url}/key-metrics/{ticker}?apikey={API_KEYS['fmp']}&limit=4"
    metrics_response = requests.get(metrics_url)
    
    # Get financial ratios
    ratios_url = f"{base_url}/ratios/{ticker}?apikey={API_KEYS['fmp']}&limit=4"
    ratios_response = requests.get(ratios_url)
    
    data = {
        'income': income_response.json() if income_response.status_code == 200 else [],
        'balance': balance_response.json() if balance_response.status_code == 200 else [],
        'cashflow': cashflow_response.json() if cashflow_response.status_code == 200 else [],
        'metrics': metrics_response.json() if metrics_response.status_code == 200 else [],
        'ratios': ratios_response.json() if ratios_response.status_code == 200 else []
    }
    
    return data

# 2. Technical Analysis Data Collection
def get_price_data(ticker, period="1y"):
    """Get historical price data using Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        # Calculate common technical indicators
        # Moving averages
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price momentum
        df['Price_Change_1M'] = df['Close'].pct_change(periods=20)
        df['Price_Change_3M'] = df['Close'].pct_change(periods=60)
        df['Price_Change_6M'] = df['Close'].pct_change(periods=125)
        df['Price_Change_1Y'] = df['Close'].pct_change(periods=252)
        
        return df
    except Exception as e:
        print(f"Error getting price data for {ticker}: {e}")
        return pd.DataFrame()

# 3. Market Sentiment Data Collection
def get_news_sentiment(ticker):
    """Get news sentiment from Marketaux"""
    base_url = "https://api.marketaux.com/v1/news/all"
    
    # Get the current date and date 7 days ago
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        'symbols': ticker,
        'filter_entities': 'true',
        'language': 'en',
        'api_token': API_KEYS['marketaux'],
        'published_after': start_date,
        'published_before': end_date
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse the news sentiment
            if 'data' in data and len(data['data']) > 0:
                sentiment_sum = 0
                article_count = len(data['data'])
                
                for article in data['data']:
                    if 'entities' in article and len(article['entities']) > 0:
                        for entity in article['entities']:
                            if entity['symbol'] == ticker:
                                sentiment_sum += entity.get('sentiment_score', 0)
                
                avg_sentiment = sentiment_sum / article_count if article_count > 0 else 0
                article_volume = article_count
                
                return {
                    'sentiment_score': avg_sentiment,
                    'article_count': article_count
                }
            
        return {
            'sentiment_score': 0,
            'article_count': 0
        }
    
    except Exception as e:
        print(f"Error getting news sentiment for {ticker}: {e}")
        return {
            'sentiment_score': 0,
            'article_count': 0
        }

# 4. Macroeconomic Data Collection
def get_macro_data():
    """Get key macroeconomic indicators from FRED"""
    # List of indicators to retrieve
    indicators = {
        'GDP': 'GDP',                          # Gross Domestic Product
        'UNRATE': 'Unemployment_Rate',         # Unemployment Rate
        'CPIAUCSL': 'CPI',                     # Consumer Price Index
        'FEDFUNDS': 'Fed_Funds_Rate',          # Federal Funds Rate
        'DFF': 'Effective_Fed_Funds_Rate',     # Effective Federal Funds Rate
        'T10Y2Y': 'Yield_Curve',               # 10-Year Treasury Minus 2-Year Treasury
        'DTWEXBGS': 'Dollar_Index',            # Trade Weighted U.S. Dollar Index
        'PAYEMS': 'Nonfarm_Payroll',           # Total Nonfarm Payroll
        'RETAILSMNSA': 'Retail_Sales',         # Retail Sales
        'REAINTRATREARAT10Y': 'Real_Interest_Rate'  # Real Interest Rate
    }
    
    macro_data = {}
    
    for code, name in indicators.items():
        try:
            series = fred_client.get_series(code)
            # Get the most recent value that isn't NaN
            latest_value = series.dropna().iloc[-1] if not series.empty else np.nan
            macro_data[name] = latest_value
        except Exception as e:
            print(f"Error getting {name} data: {e}")
            macro_data[name] = np.nan
    
    return macro_data

# 5. Industry and Sector Analysis
def get_sector_performance():
    """Get sector performance data from Financial Modeling Prep"""
    url = f"https://financialmodelingprep.com/api/v3/stock/sectors-performance?apikey={API_KEYS['fmp']}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Error getting sector performance: {e}")
        return []

# 6. Feature Engineering and Data Integration
def create_feature_set(ticker):
    """Create a comprehensive feature set for the given ticker"""
    features = {}
    
    # Get company info and current quote
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic info
        features['Symbol'] = ticker
        features['Company_Name'] = info.get('shortName', 'Unknown')
        features['Sector'] = info.get('sector', 'Unknown')
        features['Industry'] = info.get('industry', 'Unknown')
        features['Market_Cap'] = info.get('marketCap', 0)
        features['Current_Price'] = info.get('currentPrice', 0)
        features['Forward_PE'] = info.get('forwardPE', 0)
        features['PEG_Ratio'] = info.get('pegRatio', 0)
        features['Dividend_Yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        features['Beta'] = info.get('beta', 0)
        
    except Exception as e:
        print(f"Error getting basic info for {ticker}: {e}")
        features['Symbol'] = ticker
        features['Company_Name'] = 'Unknown'
        features['Sector'] = 'Unknown'
        features['Industry'] = 'Unknown'
        features['Market_Cap'] = 0
        features['Current_Price'] = 0
        features['Forward_PE'] = 0
        features['PEG_Ratio'] = 0
        features['Dividend_Yield'] = 0
        features['Beta'] = 0
    
    # Get Financial Data
    financial_data = get_financial_data(ticker)
    
    # Process income statement
    if financial_data['income'] and len(financial_data['income']) > 0:
        latest = financial_data['income'][0]
        previous = financial_data['income'][1] if len(financial_data['income']) > 1 else None
        
        features['Revenue'] = latest.get('revenue', 0)
        features['Net_Income'] = latest.get('netIncome', 0)
        features['EPS'] = latest.get('eps', 0)
        features['Gross_Profit_Margin'] = (latest.get('grossProfit', 0) / latest.get('revenue', 1)) * 100 if latest.get('revenue', 0) > 0 else 0
        features['Operating_Margin'] = (latest.get('operatingIncome', 0) / latest.get('revenue', 1)) * 100 if latest.get('revenue', 0) > 0 else 0
        features['Net_Profit_Margin'] = (latest.get('netIncome', 0) / latest.get('revenue', 1)) * 100 if latest.get('revenue', 0) > 0 else 0
        
        # Calculate growth rates
        if previous:
            features['Revenue_Growth'] = ((latest.get('revenue', 0) - previous.get('revenue', 0)) / previous.get('revenue', 1)) * 100 if previous.get('revenue', 0) > 0 else 0
            features['Net_Income_Growth'] = ((latest.get('netIncome', 0) - previous.get('netIncome', 0)) / previous.get('netIncome', 1)) * 100 if previous.get('netIncome', 0) > 0 else 0
            features['EPS_Growth'] = ((latest.get('eps', 0) - previous.get('eps', 0)) / previous.get('eps', 1)) * 100 if previous.get('eps', 0) > 0 else 0
        else:
            features['Revenue_Growth'] = 0
            features['Net_Income_Growth'] = 0
            features['EPS_Growth'] = 0
    
    # Process balance sheet
    if financial_data['balance'] and len(financial_data['balance']) > 0:
        latest = financial_data['balance'][0]
        
        features['Total_Assets'] = latest.get('totalAssets', 0)
        features['Total_Liabilities'] = latest.get('totalLiabilities', 0)
        features['Total_Debt'] = latest.get('totalDebt', 0)
        features['Cash_And_Equivalents'] = latest.get('cashAndCashEquivalents', 0)
        features['Debt_To_Equity'] = latest.get('totalDebt', 0) / (latest.get('totalAssets', 1) - latest.get('totalLiabilities', 0)) if (latest.get('totalAssets', 0) - latest.get('totalLiabilities', 0)) > 0 else 100
    
    # Process cash flow
    if financial_data['cashflow'] and len(financial_data['cashflow']) > 0:
        latest = financial_data['cashflow'][0]
        
        features['Operating_Cash_Flow'] = latest.get('operatingCashFlow', 0)
        features['Free_Cash_Flow'] = latest.get('freeCashFlow', 0)
        features['Cash_Flow_To_Debt'] = latest.get('operatingCashFlow', 0) / latest.get('totalDebt', 1) if latest.get('totalDebt', 0) > 0 else 100
    
    # Process metrics and ratios
    if financial_data['metrics'] and len(financial_data['metrics']) > 0:
        latest = financial_data['metrics'][0]
        
        features['ROE'] = latest.get('roe', 0) * 100
        features['ROA'] = latest.get('roa', 0) * 100
        features['Current_Ratio'] = latest.get('currentRatio', 0)
        features['Quick_Ratio'] = latest.get('quickRatio', 0)
        features['Interest_Coverage'] = latest.get('interestCoverage', 0)
    
    # Get technical indicators
    price_data = get_price_data(ticker)
    
    if not price_data.empty:
        latest_data = price_data.iloc[-1]
        
        features['Price_Change_1M'] = latest_data.get('Price_Change_1M', 0) * 100
        features['Price_Change_3M'] = latest_data.get('Price_Change_3M', 0) * 100
        features['Price_Change_6M'] = latest_data.get('Price_Change_6M', 0) * 100
        features['Price_Change_1Y'] = latest_data.get('Price_Change_1Y', 0) * 100
        
        features['RSI'] = latest_data.get('RSI', 50)
        features['MACD'] = latest_data.get('MACD', 0)
        features['MACD_Signal'] = latest_data.get('Signal_Line', 0)
        features['MACD_Histogram'] = latest_data.get('MACD_Histogram', 0)
        
        features['Volatility'] = latest_data.get('Volatility', 0)
        features['MA_50'] = latest_data.get('MA_50', 0)
        features['MA_200'] = latest_data.get('MA_200', 0)
        features['Golden_Cross'] = 1 if latest_data.get('MA_50', 0) > latest_data.get('MA_200', 0) else 0
    
    # Get news sentiment
    sentiment_data = get_news_sentiment(ticker)
    features['News_Sentiment'] = sentiment_data.get('sentiment_score', 0)
    features['News_Volume'] = sentiment_data.get('article_count', 0)
    
    return features

# 7. Feature Importance and Stock Selection
def rank_stocks(stocks_data):
    """Rank stocks based on a composite score of key metrics"""
    df = pd.DataFrame(stocks_data)
    
    # Define weights for different categories
    weights = {
        'Financial_Health': 0.25,
        'Growth': 0.25,
        'Valuation': 0.20,
        'Technical': 0.15,
        'Sentiment': 0.15
    }
    
    # Calculate scores for each category
    
    # Financial Health Score
    df['Financial_Health_Score'] = (
        df['Current_Ratio'].rank(pct=True) * 0.15 +
        df['Debt_To_Equity'].rank(pct=True, ascending=False) * 0.15 +  # Lower is better
        df['Interest_Coverage'].rank(pct=True) * 0.2 +
        df['Cash_Flow_To_Debt'].rank(pct=True) * 0.2 +
        df['ROE'].rank(pct=True) * 0.15 +
        df['ROA'].rank(pct=True) * 0.15
    )
    
    # Growth Score
    df['Growth_Score'] = (
        df['Revenue_Growth'].rank(pct=True) * 0.25 +
        df['Net_Income_Growth'].rank(pct=True) * 0.25 +
        df['EPS_Growth'].rank(pct=True) * 0.25 +
        df['Price_Change_6M'].rank(pct=True) * 0.15 +
        df['Operating_Cash_Flow'].rank(pct=True) * 0.10
    )
    
    # Valuation Score
    df['Valuation_Score'] = (
        df['Forward_PE'].rank(pct=True, ascending=False) * 0.3 +  # Lower is better
        df['PEG_Ratio'].rank(pct=True, ascending=False) * 0.3 +   # Lower is better
        df['Free_Cash_Flow'].rank(pct=True) * 0.2 +
        df['Dividend_Yield'].rank(pct=True) * 0.2
    )
    
    # Technical Score
    df['Technical_Score'] = (
        df['RSI'].apply(lambda x: 1 - abs((x - 50) / 50)).rank(pct=True) * 0.2 +  # Closer to 50 is better
        df['MACD_Histogram'].rank(pct=True) * 0.2 +
        df['Golden_Cross'].rank(pct=True) * 0.2 +
        df['Price_Change_1M'].rank(pct=True) * 0.2 +
        df['Volatility'].rank(pct=True, ascending=False) * 0.2    # Lower volatility is better
    )
    
    # Sentiment Score
    df['Sentiment_Score'] = (
        df['News_Sentiment'].rank(pct=True) * 0.7 +
        df['News_Volume'].rank(pct=True) * 0.3
    )
    
    # Calculate composite score
    df['Composite_Score'] = (
        df['Financial_Health_Score'] * weights['Financial_Health'] +
        df['Growth_Score'] * weights['Growth'] +
        df['Valuation_Score'] * weights['Valuation'] +
        df['Technical_Score'] * weights['Technical'] +
        df['Sentiment_Score'] * weights['Sentiment']
    )
    
    # Rank stocks by composite score
    df = df.sort_values('Composite_Score', ascending=False)
    
    return df

# 8. Main function
def main():
    print("Starting stock analysis...")
    
    # Get macro data once (since it's the same for all stocks)
    print("Fetching macroeconomic data...")
    macro_data = get_macro_data()
    print("Macroeconomic data fetched.")
    
    # Get sector performance once
    print("Fetching sector performance...")
    sector_performance = get_sector_performance()
    print("Sector performance fetched.")
    
    # Get list of stocks to analyze
    tickers = get_sp500_companies()
    print(f"Analyzing {len(tickers)} stocks...")
    
    # For demo purposes, limit to a sample of stocks
    sample_size = min(50, len(tickers))  # Analyze max 50 stocks to stay within API limits
    sample_tickers = tickers[:sample_size]
    
    all_features = []
    
    # Collect features for all stocks
    for i, ticker in enumerate(sample_tickers):
        print(f"Processing {ticker} ({i+1}/{len(sample_tickers)})...")
        
        features = create_feature_set(ticker)
        all_features.append(features)
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(1)
    
    # Rank the stocks
    print("Ranking stocks based on analysis...")
    ranked_stocks = rank_stocks(all_features)
    
    # Get the top 10 stocks
    top_stocks = ranked_stocks.head(10)
    
    print("\n===== TOP 10 RECOMMENDED STOCKS =====")
    
    stock_recommendations = []
    
    for i, (index, row) in enumerate(top_stocks.iterrows()):
        print(f"\n{i+1}. {row['Company_Name']} ({row['Symbol']})")
        print(f"   Sector: {row['Sector']}")
        print(f"   Current Price: ${row['Current_Price']:.2f}")
        print(f"   Composite Score: {row['Composite_Score']:.4f}")
        
        # Create explanation
        explanation = {
            'symbol': row['Symbol'],
            'company_name': row['Company_Name'],
            'sector': row['Sector'],
            'current_price': row['Current_Price'],
            'composite_score': row['Composite_Score'],
            'financial_strengths': [],
            'growth_potential': [],
            'technical_indicators': [],
            'risks': []
        }
        
        # Financial strengths
        if row['ROE'] > 15:
            explanation['financial_strengths'].append(f"Strong return on equity ({row['ROE']:.1f}%)")
        if row['Debt_To_Equity'] < 1:
            explanation['financial_strengths'].append(f"Low debt-to-equity ratio ({row['Debt_To_Equity']:.2f})")
        if row['Current_Ratio'] > 1.5:
            explanation['financial_strengths'].append(f"Healthy current ratio ({row['Current_Ratio']:.2f})")
        if row['Interest_Coverage'] > 5:
            explanation['financial_strengths'].append(f"Strong interest coverage ({row['Interest_Coverage']:.2f}X)")
        if row['Net_Profit_Margin'] > 10:
            explanation['financial_strengths'].append(f"Above-average profit margin ({row['Net_Profit_Margin']:.1f}%)")
        
        # Growth potential
        if row['Revenue_Growth'] > 10:
            explanation['growth_potential'].append(f"Strong revenue growth ({row['Revenue_Growth']:.1f}%)")
        if row['EPS_Growth'] > 10:
            explanation['growth_potential'].append(f"Healthy earnings growth ({row['EPS_Growth']:.1f}%)")
        if row['PEG_Ratio'] < 1.5 and row['PEG_Ratio'] > 0:
            explanation['growth_potential'].append(f"Attractive PEG ratio ({row['PEG_Ratio']:.2f})")
        if row['Forward_PE'] < 20 and row['Forward_PE'] > 0:
            explanation['growth_potential'].append(f"Reasonable forward P/E ({row['Forward_PE']:.2f})")
        
        # Technical indicators
        if row['Golden_Cross'] == 1:
            explanation['technical_indicators'].append("Positive golden cross (50-day MA above 200-day MA)")
        if 40 <= row['RSI'] <= 60:
            explanation['technical_indicators'].append(f"Neutral RSI ({row['RSI']:.1f})")
        elif row['RSI'] < 30:
            explanation['technical_indicators'].append(f"Potentially oversold (RSI: {row['RSI']:.1f})")
        if row['MACD_Histogram'] > 0:
            explanation['technical_indicators'].append("Positive MACD histogram")
        if row['News_Sentiment'] > 0:
            explanation['technical_indicators'].append(f"Positive news sentiment ({row['News_Sentiment']:.2f})")
        
        # Risks
        if row['Volatility'] > 2:
            explanation['risks'].append(f"Higher than average volatility ({row['Volatility']:.2f})")
        if row['Beta'] > 1.5:
            explanation['risks'].append(f"High beta ({row['Beta']:.2f}) indicating greater market sensitivity")
        if row['Debt_To_Equity'] > 2:
            explanation['risks'].append(f"High debt levels (D/E: {row['Debt_To_Equity']:.2f})")
        if row['RSI'] > 70:
            explanation['risks'].append(f"Potentially overbought (RSI: {row['RSI']:.1f})")
        
        stock_recommendations.append(explanation)
    
    # Save recommendations to a file
    with open('stock_recommendations.json', 'w') as f:
        json.dump(stock_recommendations, f, indent=4)
    
    print("\nAnalysis complete. Detailed results saved to stock_recommendations.json")
    return stock_recommendations

if __name__ == "__main__":
    main()