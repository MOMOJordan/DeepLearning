import pandas as pd
import numpy as np

# Sample data loading (replace this with your actual data loading code)
# Assuming the CSV has columns: 'timestamp', 'TTEF_close', 'WTI_close', 'interest_rate', 'eur_usd', 'market_sentiment', 'market_index'
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Calculate returns
data['TTEF_return'] = np.log(data['TTEF_close'] / data['TTEF_close'].shift(1))
data['WTI_return'] = np.log(data['WTI_close'] / data['WTI_close'].shift(1))

# Calculate cumulative returns over different time windows
data['TTEF_return_30min'] = np.log(data['TTEF_close'] / data['TTEF_close'].shift(30 * 60))
data['TTEF_return_2h'] = np.log(data['TTEF_close'] / data['TTEF_close'].shift(2 * 60 * 60))
data['WTI_return_30min'] = np.log(data['WTI_close'] / data['WTI_close'].shift(30 * 60))
data['WTI_return_2h'] = np.log(data['WTI_close'] / data['WTI_close'].shift(2 * 60 * 60))

# Calculate rolling volatility
data['TTEF_volatility_30min'] = data['TTEF_return'].rolling(window=30 * 60).std()
data['TTEF_volatility_1h'] = data['TTEF_return'].rolling(window=60 * 60).std()
data['WTI_volatility_30min'] = data['WTI_return'].rolling(window=30 * 60).std()
data['WTI_volatility_1h'] = data['WTI_return'].rolling(window=60 * 60).std()

# Calculate volatility ratio
data['TTEF_volatility_ratio'] = data['TTEF_volatility_30min'] / data['TTEF_volatility_1h']
data['WTI_volatility_ratio'] = data['WTI_volatility_30min'] / data['WTI_volatility_1h']

# Calculate moving averages
data['TTEF_SMA_30min'] = data['TTEF_close'].rolling(window=30 * 60).mean()
data['TTEF_EMA_30min'] = data['TTEF_close'].ewm(span=30 * 60, adjust=False).mean()
data['WTI_SMA_30min'] = data['WTI_close'].rolling(window=30 * 60).mean()
data['WTI_EMA_30min'] = data['WTI_close'].ewm(span=30 * 60, adjust=False).mean()

# Calculate RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['TTEF_RSI'] = calculate_rsi(data['TTEF_close'], window=60 * 15)  # 15 minutes
data['WTI_RSI'] = calculate_rsi(data['WTI_close'], window=60 * 15)

# Calculate Bollinger Bands
data['TTEF_BB_upper'] = data['TTEF_SMA_30min'] + 2 * data['TTEF_volatility_30min']
data['TTEF_BB_lower'] = data['TTEF_SMA_30min'] - 2 * data['TTEF_volatility_30min']

# Calculate MACD
def calculate_macd(series, slow=26, fast=12, signal=9):
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

data['TTEF_MACD'], data['TTEF_MACD_signal'] = calculate_macd(data['TTEF_close'], slow=60 * 26, fast=60 * 12, signal=60 * 9)
data['WTI_MACD'], data['WTI_MACD_signal'] = calculate_macd(data['WTI_close'], slow=60 * 26, fast=60 * 12, signal=60 * 9)

# Calculate oil price shock indicators
threshold = 0.02  # Define a threshold for significant price movements
data['WTI_shock'] = (data['WTI_return'].abs() > threshold).astype(int)

# Incorporate external macroeconomic indicators
data['interest_rate_change'] = data['interest_rate'].diff()
data['eur_usd_change'] = data['eur_usd'].diff()

# Calculate cross-asset momentum
data['TTEF_market_momentum'] = data['TTEF_return'] - data['market_index'].pct_change()
data['WTI_market_momentum'] = data['WTI_return'] - data['market_index'].pct_change()

# Calculate relative performance indicators
data['TTEF_relative_performance'] = data['TTEF_return'] - data['market_index'].pct_change()
data['WTI_relative_performance'] = data['WTI_return'] - data['market_index'].pct_change()

# Calculate macro-conditioned volatility
data['TTEF_macro_volatility'] = data['TTEF_volatility_30min'] * data['market_sentiment']
data['WTI_macro_volatility'] = data['WTI_volatility_30min'] * data['market_sentiment']

# Calculate interaction terms
data['WTI_return_vol_interaction'] = data['WTI_return'] * data['WTI_volatility_30min']

# Calculate rolling correlation
data['rolling_corr'] = data['TTEF_return'].rolling(window=60 * 60).corr(data['WTI_return'])

# Calculate spread and ratio features
data['return_spread'] = data['TTEF_return'] - data['WTI_return']
data['volatility_spread'] = data['TTEF_volatility_30min'] - data['WTI_volatility_30min']
data['return_ratio'] = data['TTEF_return'] / data['WTI_return']
data['volatility_ratio'] = data['TTEF_volatility_30min'] / data['WTI_volatility_30min']

# Drop missing values
data = data.dropna()

# Display the resulting DataFrame
print(data.head())


import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Calculate oil curve features
data['WTI_slope'] = data['CL2'] - data['CL1']
data['WTI_curve_roll'] = data['CL3'] - 2 * data['CL2'] + data['CL1']

# Calculate oil basis
data['WTI_basis'] = data['CL1'] - data['spot_oil_price']

# Calculate oil volatility skew
data['oil_vol_skew'] = data['WTI_return'].rolling(window=60*60).std() - data['WTI_return'].rolling(window=15*60).std()

# Macro-regime conditioning features
data['macro_oil_interaction'] = data['eur_usd'].pct_change() * data['WTI_return']

# Market structure / price action features
data['resid'] = data['TTEF_return_30min'] - beta * data['WTI_return_30min']
data['resid_lag1'] = data['resid'].shift(1)
data['corr_break'] = data['TTEF_return'].rolling(window=300).corr(data['WTI_return']) - data['TTEF_return'].rolling(window=3600).corr(data['WTI_return'])

# Derived structural features
data['regime'] = (data['WTI_vol_1h'] > data['WTI_vol_1h'].rolling(window=3600).median()).astype(int)
data['vol_cond_oil'] = data['WTI_return'] * data['regime']

# Interaction terms
data['oil_TTEF_interaction'] = data['WTI_return'] * data['TTEF_vol_1h']
data['oil_momentum_EURUSD'] = data['WTI_return_1h'] * data['eur_usd'].pct_change(periods=3600)

# Drop missing values
data = data.dropna()

# Display the resulting DataFrame
print(data.head())


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nolds.measures import lyap_r

# Sample data loading (replace this with your actual data loading code)
# Assuming the CSV has columns: 'timestamp', 'TTEF_open', 'TTEF_high', 'TTEF_low', 'TTEF_close', 'TTEF_volume', 'WTI_open', 'WTI_high', 'WTI_low', 'WTI_close', 'WTI_volume'
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Calculate basic returns and volatility
data['TTEF_return'] = np.log(data['TTEF_close'] / data['TTEF_close'].shift(1))
data['WTI_return'] = np.log(data['WTI_close'] / data['WTI_close'].shift(1))
data['TTEF_volatility_30min'] = data['TTEF_return'].rolling(window=30 * 60).std()

# 1. Microstructure Ghost Features
data['WTI_bidask_estimate'] = (data['WTI_high'] - data['WTI_low']) / data['WTI_close'] * 100
data['TTEF_ghost_volume'] = (data['TTEF_close'] - data['TTEF_open']).abs() / data['TTEF_volatility_30min']

# 2. Quantum Finance-Inspired Features
data['wave_corr'] = np.sin(np.pi * (data['WTI_return'].rolling(30).corr(data['TTEF_return']) + 1) / 2)

# 3. Limit Order Book Aliens (VPIN approximation)
trade_imbalance = np.where(data['WTI_close'] > data['WTI_open'], data['WTI_volume'], -data['WTI_volume'])
data['VPIN_1h'] = trade_imbalance.rolling(60).sum() / data['WTI_volume'].rolling(60).sum()

# 4. Neural Features (LSTM)
X = []
y = []
window_size = 10

# Prepare data for LSTM
for i in range(len(data) - window_size):
    X.append(data['WTI_return'].values[i:i + window_size])
    y.append(data['WTI_return'].values[i + window_size])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train LSTM model
model = Sequential([
    LSTM(8, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=5, verbose=0)

# Predict and calculate residuals
predictions = model.predict(X)
data['WTI_LSTM_residual'] = np.nan
data.iloc[window_size:, data.columns.get_loc('WTI_LSTM_residual')] = predictions.flatten() - y

# 5. Chaos Theory Features (Lyapunov exponent)
data['WTI_lyap_1h'] = data['WTI_close'].rolling(60).apply(lambda x: lyap_r(x, emb_dim=3))

# 6. Social Media Alchemy ("Dumb Money" signal)
data['WTI_dumb_money'] = ((data['WTI_return'].abs() > data['WTI_return'].std() * 2) &
                         (data['WTI_volume'] > data['WTI_volume'].mean() * 1.5)).astype(int)

# Drop missing values
data = data.dropna()

# Display the resulting DataFrame
print(data.head())

"""1. TA-Lib (Technical Analysis Library)
Strengths: TA-Lib is a comprehensive library for computing technical analysis indicators. It is widely used in the financial industry and provides a vast array of functions for calculating indicators like moving averages, RSI, MACD, and Bollinger Bands.
Considerations: It requires a C dependency, which can sometimes make installation challenging, especially on Windows systems.
Use Case: Ideal for users who need a wide range of technical indicators and are comfortable handling potential installation complexities.
2. TA (by Bukosabino)
Strengths: This is a pure-Python alternative to TA-Lib, making it easier to install and integrate into projects without dealing with C dependencies. It is built on pandas, making it very user-friendly.
Use Case: Suitable for users who need a lightweight, easy-to-install library for technical analysis without the complexity of TA-Lib.
3. Finta
Strengths: Finta is another pure-Python library that is easy to use and integrates well with pandas. It provides a variety of technical indicators and is designed to be straightforward.
Use Case: Great for users who want a simple and clean interface for generating technical indicators without extensive setup.
4. btalib
Strengths: btalib is efficient and pandas-native, making it a good choice for those who are also using the backtrader library for backtesting. It is designed to work seamlessly with backtrader.
Use Case: Best for users who are already using backtrader for backtesting and want an integrated solution for technical analysis.
5. Featuretools
Strengths: Featuretools is designed for automated feature engineering and is particularly useful for generating features from tabular datasets. It can automatically create lag features, aggregations, and rolling windows.
Use Case: Ideal for users who need to automate feature engineering and are working with structured datasets. It can be adapted for time series data as well.
6. tsfresh
Strengths: tsfresh is designed to extract a large number of time-series features automatically. It can compute features like FFT coefficients, autocorrelation, skew, and more, making it useful for exploratory data analysis.
Use Case: Suitable for users who are unsure about which signals to look for and want a brute-force approach to feature extraction.
Bonus Libraries
arch: For GARCH volatility modeling, useful for users who need to model and forecast financial volatility.
statsmodels: Provides a wide range of statistical models, including ARIMA, OLS, and cointegration tests, useful for econometric analysis.
pyfolio: For portfolio analytics, useful for users who need to analyze the performance of investment portfolios.
ffn and empyrical: For financial metrics and return/risk statistics, useful for users who need to compute performance metrics and risk statistics.
Recommendations
Classic Technical Analysis: Use TA-Lib or TA for a comprehensive set of technical indicators.
No External Dependencies: Use TA or Finta for pure-Python solutions.
Auto-Discovery of Features: Use tsfresh for extracting a wide range of time-series features automatically.
Quantitative Analysis and Backtesting: Use btalib with backtrader for integrated backtesting solutions.
Tabular Machine Learning: Use Featuretools for automated feature engineering on tabular data."""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna

# Load your data
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Calculate the target variable, e.g., 1-hour future return
data['TTEF_target'] = data['TTEF_close'].pct_change(periods=60).shift(-60)

# Drop rows where target is NaN (due to shift)
data = data.dropna(subset=['TTEF_target'])

# Generate all TA features
data = add_all_ta_features(data, open="TTEF_open", high="TTEF_high", low="TTEF_low", close="TTEF_close", volume="TTEF_volume")

# Drop rows with NaN values resulting from TA feature calculations
data = dropna(data)

# Calculate correlation with the target variable
correlations = data.corrwith(data['TTEF_target']).sort_values(ascending=False)

# Display the top features with the highest correlation
print(correlations.head(20))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data with a non-linear relationship
np.random.seed(0)
x = np.linspace(-10, 10, 1000)
y = 0.5 * x**2 - 3 * x + np.random.normal(0, 10, 1000)  # Quadratic relationship with noise

# Create a DataFrame
data = pd.DataFrame({'Feature': x, 'Target': y})

# Rank the feature into quantiles (e.g., deciles)
data['Feature_Quantile'] = pd.qcut(data['Feature'], q=10, duplicates='drop')

# Calculate the mean of the target variable for each quantile
quantile_means = data.groupby('Feature_Quantile')['Target'].mean().reset_index()

# Convert quantile labels to a numerical format for plotting
quantile_means['Quantile_Label'] = quantile_means['Feature_Quantile'].apply(lambda x: int(x.right.split(',')[0].replace('(', '')))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(quantile_means['Quantile_Label'], quantile_means['Target'], marker='o')
plt.title('Mean of Target Variable by Feature Quantile')
plt.xlabel('Feature Quantile')
plt.ylabel('Mean of Target Variable')
plt.xticks(quantile_means['Quantile_Label'], labels=[f"Q{i+1}" for i in range(len(quantile_means))])
plt.grid(True)
plt.show()

1. “The Time-Dependent Lead–Lag Relationship Between WTI and Brent Crude Oil Spot Markets”
Frontiers in Physics (2020)

Applies the Thermal Optimal Path (TOPS) method to daily oil data from 1987–2017.

Demonstrates a dynamic, event-conditioned lead–lag structure, with WTI often leading Brent, especially around crises 
public.econ.duke.edu
+15
frontiersin.org
+15
sciencedirect.com
+15
.

2. “The Lead–Lag Relationship Between Spot and Futures Markets in Energy Sector Stocks”
International Journal of Energy Economics and Policy (2020)

Uses 1-minute intraday data for stock-specific energy futures vs. spot.

Finds lead–lag effects lasting up to 30 minutes, with intraday high-frequency patterns 
snf.no
+2
econjournals.com
+2
researchgate.net
+2
.

3. “Lead Lat Relationships Between Futures and Spot Prices”
A PDF exploration of cointegration and Johansen multivariate analysis across commodity spot and futures.

Useful for understanding policy-driven infections of lead dynamics 
snf.no
+1
econjournals.com
+1
arxiv.org
+1
research.cbs.dk
+1
.

4. “Testing the Relationship Between Oil Equities and Oil Futures”
Duke Univ. Honors Thesis (2008)

Studies oil futures vs. oil-equity returns correlation and volatility spillover.

Focuses on jump behavior, proving correlation but also highlighting differences in vol dynamics 
dergipark.org.tr
+5
public.econ.duke.edu
+5
link.springer.com
+5
.


