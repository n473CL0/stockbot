import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    """Fetch OHLCV stock price data with safe fallback and flattened columns."""

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

        if df is None or df.empty:
            st.error(f"❌ No stock data found for {ticker}.")
            return None

        # --- Flatten MultiIndex columns ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Reset index so Date is a column
        df = df.reset_index()

        # --- Detect Close column ---
        close_col = next((c for c in df.columns if "Close" in c), None)
        if close_col is None:
            st.error(f"❌ No Close column found. Columns: {list(df.columns)}")
            return None
        df["Close"] = df[close_col]

        # --- Standardize Date ---
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.dropna(subset=["Date"], inplace=True)
        df["DateOnly"] = df["Date"].dt.normalize()

        # --- Add Pct_Change column ---
        df["Pct_Change"] = df["Close"].pct_change() * 100

        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch stock data: {e}")
        return None

    






