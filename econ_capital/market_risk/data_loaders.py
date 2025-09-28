def load_real_risk_factors(start="2020-01-01", end="2025-01-01") -> pd.DataFrame:
    tickers = {
        "SPY": "Equities (US)",
        "EFA": "Equities (Developed ex-US)",
        "EEM": "Equities (EM)",
        "TLT": "US Treasuries (long duration)",
        "LQD": "IG Credit ETF",
        "HYG": "HY Credit ETF",
        "GLD": "Gold",
        "USO": "Oil",
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
    }
    data = yf.download(list(tickers.keys()), start=start, end=end)
    if "Adj Close" in data:
        prices = data["Adj Close"].dropna()
    else:
        prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()
    returns.columns = list(tickers.keys())
    return returns

def load_dummy_positions_real() -> pd.DataFrame:
    idx = ["Equity_US", "Equity_EM", "Rates", "Credit_IG", "Credit_HY", "Gold", "Oil", "FX_EURUSD", "FX_GBPUSD"]
    df = pd.DataFrame(index=idx)
    df["SPY"] = [5_000_000, 0, 0, 0, 0, 0, 0, 0, 0]
    df["EEM"] = [0, 2_000_000, 0, 0, 0, 0, 0, 0, 0]
    df["TLT"] = [0, 0, -10_000.0, 0, 0, 0, 0, 0, 0]
    df["LQD"] = [0, 0, 0, -3_000.0, 0, 0, 0, 0, 0]
    df["HYG"] = [0, 0, 0, 0, -4_000.0, 0, 0, 0, 0]
    df["GLD"] = [0, 0, 0, 0, 0, 1_000_000, 0, 0, 0]
    df["USO"] = [0, 0, 0, 0, 0, 0, 500_000, 0, 0]
    df["EURUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 2_000_000, 0]
    df["GBPUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 0, 1_500_000]
    return df.fillna(0.0)