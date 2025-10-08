import pandas as pd
import numpy as np
import streamlit as st
import time
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- Helper Indicators ----------
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    avg_up = up.rolling(window, min_periods=1).mean()
    avg_down = down.rolling(window, min_periods=1).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ---------- Main Forecast Function ----------
def fit_and_forecast(combined_df: pd.DataFrame, steps: int = 7):
    try:
        df = combined_df.copy()
        print(f"🔧 fit_and_forecast received {len(df)} rows with columns: {sorted(df.columns.tolist())}")

        # Ensure datetime index
        if "DateOnly" in df.columns:
            df["DateOnly"] = pd.to_datetime(df["DateOnly"]).dt.normalize()
            df = df.set_index("DateOnly")

        required_cols = {"Pct_Change", "Close", "sentiment_score"}
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            msg = f"❌ Missing required columns for forecasting: {missing}"
            print(msg)
            st.error(msg)
            return None

        df = df.sort_index()

        # ---------- Feature Engineering ----------
        df["Volatility"] = df["Pct_Change"].rolling(7, min_periods=1).std().fillna(0)
        if "Volume" in df.columns:
            df["LogVolume"] = np.log1p(df["Volume"].clip(lower=0))
        else:
            df["LogVolume"] = 0.0
        df["RSI"] = compute_rsi(df["Close"]).clip(lower=0, upper=100)
        macd, signal = compute_macd(df["Close"].fillna(method="ffill").fillna(method="bfill"))
        df["MACD"], df["MACD_signal"] = macd.fillna(0.0), signal.fillna(0.0)
        df["MA20"] = df["Close"].rolling(20, min_periods=1).mean().fillna(method="bfill").fillna(method="ffill")
        df["MA50"] = df["Close"].rolling(50, min_periods=1).mean().fillna(method="bfill").fillna(method="ffill")

        # Add VIX
        try:
            vix = yf.download("^VIX", start=df.index.min(), end=df.index.max(), progress=False)["Close"]
            df["VIX"] = vix.reindex(df.index).ffill()
        except Exception as e:
            warn_msg = f"⚠️ VIX fetch failed: {e}"
            print(warn_msg)
            st.warning(warn_msg)
            df["VIX"] = 0.0

        exog_cols = ["sentiment_score","Volatility","LogVolume","RSI","MACD","MACD_signal","MA20","MA50","VIX"]
        exog = df[exog_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        y = df["Pct_Change"].replace([np.inf, -np.inf], np.nan).dropna()
        exog = exog.loc[y.index]

        if y.empty or y.nunique() < 2:
            st.error("❌ Not enough variation in target to forecast.")
            return None

        # ---------- Fit SARIMAX ----------
        st.info(f"📊 Training SARIMAX on {len(y)} samples with {len(exog_cols)} exogenous vars...")
        t0 = time.time()
        print(f"🔧 Training SARIMAX with exog shape {exog.shape} and target length {len(y)}")
        model = SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(1,0,1,5),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        elapsed = time.time() - t0
        st.success(f"⏱️ SARIMAX fit took {elapsed:.2f}s")

        # ---------- Forecast ----------
        future_exog = exog.iloc[-1:].copy()
        future_exog = pd.concat([future_exog]*steps, ignore_index=True)
        print(f"🔧 Forecasting next {steps} steps using future_exog shape {future_exog.shape}")
        forecast_res = model_fit.get_forecast(steps=steps, exog=future_exog)

        forecast_mean = forecast_res.predicted_mean
        forecast_ci = forecast_res.conf_int()

        forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        forecast_df = pd.DataFrame({"Forecast": forecast_mean.values}, index=forecast_index)
        forecast_df["Lower"], forecast_df["Upper"] = forecast_ci.iloc[:,0].values, forecast_ci.iloc[:,1].values

        # ---------- Weekly aggregation ----------
        weekly_forecast = forecast_df.resample("W").sum()

        # ---------- Buy / Sell / Hold ----------
        avg_return = forecast_df["Forecast"].mean()
        certainty = (forecast_df["Forecast"] > 0).mean() if avg_return > 0 else (forecast_df["Forecast"] < 0).mean()

        if avg_return > 0.5 and certainty > 0.6:
            rec = "BUY"
        elif avg_return < -0.5 and certainty > 0.6:
            rec = "SELL"
        else:
            rec = "HOLD"

        result_payload = {
            "daily_forecast": forecast_df,
            "weekly_forecast": weekly_forecast,
            "certainty": certainty,
            "recommendation": rec
        }
        print("✅ Forecast generated successfully")
        return result_payload

    except Exception as e:
        err_msg = f"❌ Forecasting failed: {e}"
        print(err_msg)
        st.error(err_msg)
        return None
