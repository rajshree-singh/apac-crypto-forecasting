Reproduction package (APFM version)
Run from the code/ folder, in this order:
  python prep.py            # builds panel/ from apdata/ (raw exchange, Yahoo and FRED downloads)
  python run_arima.py <series...>; python run_linear.py; python run_main.py <series...>
  python run_robust.py ablation lags hyper seeds transfer
  python premium.py; python evaluate.py; python eval_robust.py; python econ.py; python figures.py
Series: BTC-USD ETH-USD LTC-USD BTC-KRW ETH-KRW BTC-JPY ETH-JPY LTC-JPY BTC-INR ETH-INR LTC-INR
Raw data: Upbit /v1/candles/days (KRW), bitbank public candlestick 1day (JPY), WazirX /sapi/v1/klines 1d (INR),
Yahoo Finance via yfinance (USD; files BTC.csv ETH.csv LTC.csv), FRED DEXKOUS DEXJPUS DEXINUS. Downloaded 15 Oct 2026.
Packages: python 3.13, pandas, numpy, statsmodels, scikit-learn, xgboost, torch (CPU), matplotlib.
