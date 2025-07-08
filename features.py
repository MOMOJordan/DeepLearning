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

1. High-Frequency Lead-Lag Analysis
Key Papers:
"The Lead–Lag Relationship Between Spot and Futures Markets in Energy Sector Stocks"

Focus: Intraday (1-minute) analysis of energy futures vs. spot prices.

Key Findings:

Futures lead equities by 5–30 minutes, with strongest effects during market openings.

Liquidity and order flow imbalances drive lead-lag dynamics.

Feature Ideas:

Rolling cross-correlations (e.g., 10-min windows) to detect adaptive lags.

Intraday seasonality adjustments (e.g., stronger leads at NYMEX open) 15.

"Microstructure Noise and Realized Volatility in Oil Futures"

Focus: High-frequency noise filtering for WTI futures.

Feature Ideas:

Bid-ask spread adjustments to mitigate microstructure noise.

Realized volatility (10-min rolling std. dev.) as a leading indicator 7.

2. Dynamic Lead-Lag Methods
Key Papers:
"The Time-Dependent Lead–Lag Relationship Between WTI and Brent"

Method: Thermal Optimal Path (TOPS) to detect regime-specific lags.

Key Findings:

WTI leads Brent by 1–2 days during supply shocks (e.g., 2008 crisis).

Feature Ideas:

Event-driven lag features (e.g., OPEC announcements → 1-hour lag window).

Rolling cointegration residuals for structural breaks 713.

"Dynamic Lead-Lag Networks in Commodity Markets"

Method: Wavelet coherence + Granger causality networks.

Feature Ideas:

Multi-scale leads (e.g., WTI leads TTEF at 10-min but lags at 1-hour).

Network centrality metrics to quantify WTI’s influence 7.

3. Volatility Spillovers & Jumps
Key Papers:
"Testing the Relationship Between Oil Equities and Oil Futures"

Focus: Volatility spillovers and jump dynamics.

Key Findings:

WTI jumps (>2σ) predict TTEF returns within 15 minutes.

Feature Ideas:

Jump indicators (binary flags for extreme returns).

Volatility ratios (WTI vol / TTEF vol) 113.

"Crude Oil and Stock Markets: Causal Relationships in Tail"

Method: Quantile regression for extreme events.

Feature Ideas:

Tail dependence metrics (e.g., 5% quantile co-movements) 13.

4. Machine Learning & Hybrid Models
Key Papers:
"Forecasting the WTI Crude Oil Price by a Hybrid-Refined Method"

Method: Combines PPM change-point detection with TVTP-MRS models.

Key Findings:

Structural breaks (e.g., financial crises) alter lead-lag relationships.

Feature Ideas:

Regime-switching indicators (e.g., Markov-switching residuals) 7.

"Predicting the Price of WTI Crude Oil Using ANN and Chaos"

Method: Chaos theory + ANN (HWP-CHAOS model).

Key Findings:

WTI exhibits chaotic properties (Lyapunov exponent > 0).

Feature Ideas:

Phase-space reconstruction (embedding dimension = 5, time delay = 1) 13.

"Volatility Forecasting of Crude Oil Futures Based on Bi-LSTM-Attention"

Method: Bi-LSTM with attention for event-driven volatility.

Key Findings:

Attention mechanism improves prediction during crises (e.g., COVID-19).

Feature Ideas:

News sentiment scores (e.g., COVID-19/geopolitical risk indices) 16.

5. Practical Feature Extraction Guide
Feature Type	Calculation	Source Paper
Lagged WTI Returns	WTI_ret.shift(3) (30-min lag)	17
Rolling Momentum	WTI_ret.rolling(5).mean()	13
VECM Residuals	Cointegration deviations (WTI vs. TTEF)	7
Jump Indicators	1 if abs(WTI_ret) > 2*std_1day	13
Wavelet Coherence	Multi-scale CCF (pywt library)	713


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('your_dataset.csv')

# Assume the last column is the target and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Calculate performance metrics for the linear model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# LightGBM Model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': 0
}

lgbm_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)
y_pred_lgbm = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

# Calculate performance metrics for the LightGBM model
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

# Print performance metrics
print("Linear Model - Mean Squared Error:", mse_linear)
print("Linear Model - R-squared:", r2_linear)
print("LightGBM Model - Mean Squared Error:", mse_lgbm)
print("LightGBM Model - R-squared:", r2_lgbm)

# Residual analysis for Linear Regression
residuals_linear = y_test - y_pred_linear

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_linear, y=residuals_linear)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Linear Model')

plt.subplot(1, 2, 2)
sns.histplot(residuals_linear, kde=True)
plt.title('Distribution of Residuals for Linear Model')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()

# Residual analysis for LightGBM
residuals_lgbm = y_test - y_pred_lgbm

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_lgbm, y=residuals_lgbm)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for LightGBM Model')

plt.subplot(1, 2, 2)
sns.histplot(residuals_lgbm, kde=True)
plt.title('Distribution of Residuals for LightGBM Model')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()

# SHAP value analysis for LightGBM
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title('SHAP Summary Plot for LightGBM Model')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance for LightGBM Model')
plt.tight_layout()
plt.show()

# Compare the performance of the models
models = ['Linear Regression', 'LightGBM']
mse_scores = [mse_linear, mse_lgbm]
r2_scores = [r2_linear, r2_lgbm]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=mse_scores)
plt.title('Mean Squared Error Comparison')

plt.subplot(1, 2, 2)
sns.barplot(x=models, y=r2_scores)
plt.title('R-squared Comparison')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.stattools import adfuller

# Load data
data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Select variables
df = data[['oil_price', 'ttef_price']].dropna()

# Check for unit roots (non-stationarity)
def adf_check(series, name):
    result = adfuller(series.dropna())
    print(f'{name} ADF stat: {result[0]:.3f}, p-value: {result[1]:.3f}')

adf_check(df['oil_price'], 'Oil')
adf_check(df['ttef_price'], 'TTEF')

# Select lag order (use differences automatically)
lag_order = select_order(df, maxlags=10, deterministic="ci").selected_orders['aic']
print("Selected lag order:", lag_order)

# Select cointegration rank (Johansen test)
coint_rank = select_coint_rank(df, det_order=0, k_ar_diff=lag_order).rank
print("Selected cointegration rank:", coint_rank)

# Fit VECM model
vecm_model = VECM(df, k_ar_diff=lag_order, coint_rank=coint_rank, deterministic='ci')
vecm_result = vecm_model.fit()

# Error Correction Term (ECT): cointegration residuals (long-run mispricing)
data['error_correction_term'] = vecm_result.resid.iloc[:, 0]

# Lagged differenced variables (short-run dynamics)
for i in range(lag_order):
    data[f'doil_diff_lag{i+1}'] = df['oil_price'].diff().shift(i+1)
    data[f'dttef_diff_lag{i+1}'] = df['ttef_price'].diff().shift(i+1)

# Clean final feature set
features = data[[
    'error_correction_term',
    *[f'doil_diff_lag{i+1}' for i in range(lag_order)],
    *[f'dttef_diff_lag{i+1}' for i in range(lag_order)]
]].dropna()

print(features.head())

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class IntradayFeatureGenerator:
    def __init__(self, data, freq='1s'):
        """
        Initialize with a DataFrame containing:
        - TTEF_price, CL_price, LCO_price, GN_price
        - Optional: TTEF_high, TTEF_low, etc. for range features
        """
        self.data = data.copy()
        self.freq = freq
        self.set_time_params()
        
    def set_time_params(self):
        """Convert time frequencies to rolling window sizes"""
        self.time_mapping = {
            '10s': 10,
            '30s': 30,
            '1min': 60,
            '5min': 300,
            '10min': 600,
            '30min': 1800,
            '1h': 3600
        }
    
    def generate_all_features(self):
        """Generate complete feature set"""
        self.calculate_returns()
        self.generate_volatility_features()
        self.generate_momentum_features()
        self.generate_spread_features()
        self.generate_correlation_features()
        self.generate_regime_features()
        self.generate_zscore_features()
        self.generate_interaction_features()
        self.generate_shock_features()
        self.clean_features()
        return self.data
    
    def calculate_returns(self):
        """Calculate log returns for all assets"""
        assets = ['TTEF', 'CL', 'LCO', 'GN']
        for asset in assets:
            self.data[f'{asset}_return'] = np.log(self.data[f'{asset}_price'] / self.data[f'{asset}_price'].shift(1))
            
            # Multi-horizon returns
            for period in ['10s', '1min', '5min', '10min']:
                window = self.time_mapping[period]
                self.data[f'{asset}_return_{period}'] = np.log(self.data[f'{asset}_price'] / self.data[f'{asset}_price'].shift(window))
    
    def generate_volatility_features(self):
        """Create various volatility measures"""
        assets = ['TTEF', 'CL', 'LCO', 'GN']
        for asset in assets:
            # Standard volatility
            for period in ['1min', '5min', '10min']:
                window = self.time_mapping[period]
                self.data[f'{asset}_vol_{period}'] = self.data[f'{asset}_return'].rolling(window).std()
            
            # EWMA volatility
            self.data[f'{asset}_vol_ewma_10min'] = self.data[f'{asset}_return'].ewm(span=600).std()
            
            # Range volatility (if high/low available)
            if f'{asset}_high' in self.data.columns:
                self.data[f'{asset}_range_10min'] = (self.data[f'{asset}_high'].rolling(600).max() - 
                                                   self.data[f'{asset}_low'].rolling(600).min())
            
            # Volatility ratios
            self.data[f'{asset}_vol_ratio_1m_5m'] = (self.data[f'{asset}_vol_1min'] / 
                                                    self.data[f'{asset}_vol_5min'])
            
            # Vol of vol
            self.data[f'{asset}_vol_of_vol'] = self.data[f'{asset}_vol_1min'].diff().rolling(300).std()
    
    def generate_momentum_features(self):
        """Create trend and momentum indicators"""
        assets = ['CL', 'LCO', 'GN']
        for asset in assets:
            # EMA cross features
            self.data[f'{asset}_ema_5min'] = self.data[f'{asset}_price'].ewm(span=300).mean()
            self.data[f'{asset}_ema_30min'] = self.data[f'{asset}_price'].ewm(span=1800).mean()
            self.data[f'{asset}_momentum'] = self.data[f'{asset}_ema_5min'] - self.data[f'{asset}_ema_30min']
            
            # Lagged returns
            for lag in ['10s', '30s', '1min', '5min']:
                window = self.time_mapping[lag]
                self.data[f'{asset}_return_lag_{lag}'] = self.data[f'{asset}_return'].shift(window)
    
    def generate_spread_features(self):
        """Generate spread between commodities"""
        # Price spreads
        self.data['CL_LCO_spread'] = self.data['CL_price'] - self.data['LCO_price']
        self.data['CL_GN_spread'] = self.data['CL_price'] - self.data['GN_price']
        
        # Spread changes
        self.data['CL_LCO_spread_change'] = self.data['CL_LCO_spread'].diff()
        self.data['CL_GN_spread_change'] = self.data['CL_GN_spread'].diff()
        
        # Normalized spreads
        self.data['CL_LCO_spread_z'] = (self.data['CL_LCO_spread'] - 
                                      self.data['CL_LCO_spread'].rolling(3600).mean()) / \
                                     self.data['CL_LCO_spread'].rolling(3600).std()
    
    def generate_correlation_features(self):
        """Rolling correlations between assets"""
        windows = {'30min': 1800, '1h': 3600}
        for window_name, window in windows.items():
            self.data[f'corr_TTEF_CL_{window_name}'] = self.data['TTEF_return'].rolling(window).corr(self.data['CL_return'])
            self.data[f'corr_TTEF_LCO_{window_name}'] = self.data['TTEF_return'].rolling(window).corr(self.data['LCO_return'])
            self.data[f'corr_CL_LCO_{window_name}'] = self.data['CL_return'].rolling(window).corr(self.data['LCO_return'])
        
        # Rolling betas
        self.data['beta_TTEF_CL'] = self.data['TTEF_return'].rolling(3600).cov(self.data['CL_return']) / \
                                   self.data['CL_return'].rolling(3600).var()
    
    def generate_regime_features(self):
        """Create market regime indicators"""
        # Volatility regime
        self.data['CL_vol_regime'] = (self.data['CL_vol_10min'] > 
                                     self.data['CL_vol_10min'].rolling(3600).mean()).astype(int)
        
        # Trend regime
        self.data['CL_trend_regime'] = (self.data['CL_momentum'] > 0).astype(int)
        
        # Combined regime
        self.data['CL_highvol_uptrend'] = self.data['CL_vol_regime'] & self.data['CL_trend_regime']
    
    def generate_zscore_features(self):
        """
        Create Z-scores with different windows for mean and std
        This helps detect short-term deviations from longer-term behavior
        """
        assets = ['CL', 'LCO', 'GN']
        for asset in assets:
            # Short mean (1min), long std (30min)
            self.data[f'{asset}_z_1m_mean_30m_std'] = (
                (self.data[f'{asset}_return'] - self.data[f'{asset}_return'].rolling(60).mean()) / 
                self.data[f'{asset}_return'].rolling(1800).std()
            )
            
            # Medium mean (5min), very long std (1h)
            self.data[f'{asset}_z_5m_mean_1h_std'] = (
                (self.data[f'{asset}_return'] - self.data[f'{asset}_return'].rolling(300).mean()) / 
                self.data[f'{asset}_return'].rolling(3600).std()
            )
            
            # Volatility z-score
            self.data[f'{asset}_vol_z'] = (
                (self.data[f'{asset}_vol_5min'] - self.data[f'{asset}_vol_5min'].rolling(1800).mean()) / 
                self.data[f'{asset}_vol_5min'].rolling(1800).std()
            )
    
    def generate_interaction_features(self):
        """Create interaction terms between features"""
        self.data['CL_return_vol_interaction'] = self.data['CL_return'] * self.data['CL_vol_5min']
        self.data['CL_LCO_spread_vol_interaction'] = self.data['CL_LCO_spread'] * self.data['TTEF_vol_5min']
        self.data['CL_momentum_regime_interaction'] = self.data['CL_momentum'] * self.data['CL_vol_regime']
    
    def generate_shock_features(self):
        """Identify shock events"""
        for asset in ['CL', 'LCO', 'GN']:
            # Return shocks (volatility-adjusted)
            self.data[f'{asset}_return_shock'] = (
                (self.data[f'{asset}_return'].abs() > 
                 2 * self.data[f'{asset}_return'].rolling(600).std()).astype(int)
            
            # Volatility shocks
            vol_diff = self.data[f'{asset}_vol_5min'].diff()
            self.data[f'{asset}_vol_shock'] = (
                (vol_diff > 2 * vol_diff.rolling(1800).std()).astype(int)
    
    def clean_features(self):
        """Handle missing values and infinite values"""
        # Drop initial NaN values from rolling calculations
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(0, inplace=True)
        
        # Standardize features (optional)
        #scaler = StandardScaler()
        #features_to_scale = [col for col in self.data.columns if 'return' not in col and 'price' not in col]
        #self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])


# Example usage
if __name__ == "__main__":
    # Load your data (assuming it has columns: TTEF_price, CL_price, LCO_price, GN_price)
    # df = pd.read_csv('your_intraday_data.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # For demonstration, create mock data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=86400, freq='S')
    mock_data = {
        'TTEF_price': 100 + np.cumsum(np.random.normal(0, 0.001, 86400)),
        'CL_price': 75 + np.cumsum(np.random.normal(0, 0.002, 86400)),
        'LCO_price': 78 + np.cumsum(np.random.normal(0, 0.0018, 86400)),
        'GN_price': 3.5 + np.cumsum(np.random.normal(0, 0.0005, 86400)),
        'TTEF_high': 100 + np.cumsum(np.random.normal(0.001, 0.0015, 86400)),
        'TTEF_low': 100 + np.cumsum(np.random.normal(-0.001, 0.0015, 86400))
    }
    df = pd.DataFrame(mock_data, index=dates)
    
    # Generate features
    generator = IntradayFeatureGenerator(df)
    feature_df = generator.generate_all_features()
    
    # Display feature summary
    print(f"Generated {len(feature_df.columns)} features:")
    print(feature_df.iloc[-10:, -10:].head())  # Show last 10 features for last 10 timesteps
    
    # Optional: Save to CSV
    # feature_df.to_csv('TTEF_intraday_features.csv')
