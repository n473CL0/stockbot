# data/preprocessing.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

from utils.helpers import (
    compute_weighted_daily_sentiment,
    normalize_sentiment_columns,
    safe_utc_localize,
)


def process_sentiment_data(
    articles_df: pd.DataFrame,
    decay: float = 0.95,
    min_days: int = 15,
    debug: bool = True,
) -> pd.DataFrame:
    """Convert article-level sentiment into daily weighted sentiment."""
    if articles_df is None or articles_df.empty:
        st.warning("No sentiment data available for processing.")
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    df = articles_df.copy()
    if debug:
        st.write(f"🔍 process_sentiment_data: starting with {len(df)} articles")
        st.write("🔍 sample article columns:", df.columns.tolist())

    # --- Ensure numeric sentiment_score ---
    if "sentiment_score" not in df.columns:
        if "sentiment" in df.columns:
            st.warning("⚠️ 'sentiment_score' missing — coercing from 'sentiment'")
            df["sentiment_score"] = pd.to_numeric(df["sentiment"], errors="coerce")
        else:
            st.error("❌ Missing 'sentiment_score' and no fallback column.")
            return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    df["sentiment_score"] = (
        pd.to_numeric(df["sentiment_score"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    # --- Date handling (support Date, DateOnly, fallback to published) ---
    if "Date" in df.columns:
        df["Date"] = safe_utc_localize(df["Date"])
    elif "DateOnly" in df.columns:
        df["DateOnly"] = pd.to_datetime(df["DateOnly"], errors="coerce").dt.normalize()
        df["Date"] = safe_utc_localize(df["DateOnly"])
    elif "published" in df.columns:
        st.warning("⚠️ Using 'published' as date source")
        df["Date"] = safe_utc_localize(df["published"])
    else:
        st.error("❌ No date column found in sentiment data")
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    # Drop missing
    before = len(df)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    after = len(df)
    if debug:
        st.write(f"🔍 DEBUG: dropped {before - after} rows with invalid dates")

    # --- Daily weighted sentiment ---
    daily = compute_weighted_daily_sentiment(
        df, decay_val=decay, min_days=min_days, debug=debug
    )

    return daily


def combine_data(
    sentiment_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    lag_days: int = 1,
    join_type: str = "left",
) -> pd.DataFrame:
    """Merge stock and sentiment data while protecting stock columns."""
    if stock_df is None or sentiment_df is None or stock_df.empty:
        st.error("❌ Missing stock or sentiment data.")
        return pd.DataFrame()

    stock = stock_df.copy()
    sent = sentiment_df.copy()

    # --- Stock date ---
    if "Date" not in stock.columns and "date" in stock.columns:
        stock["Date"] = stock["date"]
    stock["Date"] = safe_utc_localize(stock.get("Date", pd.NaT))
    stock["DateOnly"] = stock["Date"].dt.tz_localize(None).dt.normalize()

    # --- Canonical Close ---
    # Try common variants, fallback to first price-like column
    close_candidates = [
        "Close", "close", "Adj Close", "Adj_Close", "adjclose"
    ]
    close_col = next((c for c in close_candidates if c in stock.columns), None)
    if close_col is not None:
        stock["Close"] = pd.to_numeric(stock[close_col], errors="coerce")
    elif any(c.lower().startswith("close") for c in stock.columns):
        c = next(c for c in stock.columns if c.lower().startswith("close"))
        stock["Close"] = pd.to_numeric(stock[c], errors="coerce")
    else:
        st.error("❌ No close price found in stock DataFrame.")
        return pd.DataFrame()

    # --- Canonical Volume (optional but helpful) ---
    vol_candidates = ["Volume", "volume", "Vol", "vol"]
    vol_col = next((c for c in vol_candidates if c in stock.columns), None)
    if vol_col is not None:
        stock["Volume"] = pd.to_numeric(stock[vol_col], errors="coerce")

    # --- Sentiment dates ---
    if "DateOnly" in sent.columns:
        sent["DateOnly"] = pd.to_datetime(sent["DateOnly"], errors="coerce").dt.normalize()
    elif "Date" in sent.columns:
        sent["DateOnly"] = pd.to_datetime(sent["Date"], errors="coerce").dt.normalize()
    else:
        return pd.DataFrame()

    # --- Canonical sentiment ---
    if "computed_score" not in sent.columns:
        sent["computed_score"] = pd.to_numeric(
            sent.get("sentiment_score", 0.0), errors="coerce"
        )
    sent["sentiment_score"] = sent["computed_score"]

    # --- Pct Change ---
    if "Pct_Change" not in stock.columns:
        stock["Pct_Change"] = stock["Close"].pct_change() * 100.0

    # --- Merge ---
    merged = pd.merge(
        stock,
        sent[["DateOnly", "computed_score", "sentiment_score"]],
        on="DateOnly",
        how=join_type,
    )
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Fill + enforce consistency
    merged["computed_score"] = merged["computed_score"].fillna(0.0)
    merged["sentiment_score"] = merged["computed_score"]
    merged["Pct_Change"] = pd.to_numeric(merged["Pct_Change"], errors="coerce").fillna(0.0)

    # --- Safeguard swap ---
    sent_abs_max = merged["computed_score"].abs().max() if not merged["computed_score"].empty else 0.0
    pct_abs_max = merged["Pct_Change"].abs().max() if not merged["Pct_Change"].empty else 0.0
    if sent_abs_max > 2.0 and pct_abs_max <= 2.0:
        st.warning("⚠️ Misalignment detected: swapping sentiment and Pct_Change.")
        tmp = merged["computed_score"].copy()
        merged["computed_score"] = merged["Pct_Change"].fillna(0.0)
        merged["sentiment_score"] = merged["computed_score"]
        merged["Pct_Change"] = tmp.fillna(0.0)

    # --- Features ---
    merged["Lagged_Sentiment"] = merged["computed_score"].shift(lag_days)
    merged["is_positive"] = (merged["computed_score"] > 0).astype(int)
    merged["7day_pct_positive"] = (
        merged["is_positive"].rolling(7, min_periods=1).mean() * 100
    )

    # --- Debug check ---
    if "Close" not in merged.columns:
        st.error("❌ Final merged DataFrame missing 'Close' column.")
    else:
        st.write("✅ combine_data produced 'Close' column with dtype:", merged["Close"].dtype)

    return merged




def prepare_sentiment_for_forecast(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Prepare daily sentiment for forecasting (no stock merge)."""
    if daily_sentiment is None or daily_sentiment.empty:
        return pd.DataFrame(columns=["DateOnly", "computed_score"])

    df = daily_sentiment.copy()
    if "DateOnly" not in df.columns and "Date" in df.columns:
        df["DateOnly"] = safe_utc_localize(df["Date"]).dt.normalize()

    df["DateOnly"] = safe_utc_localize(df["DateOnly"]).dt.normalize()
    df = df.dropna(subset=["DateOnly"]).sort_values("DateOnly").reset_index(drop=True)

    return df[["DateOnly", "computed_score"]].set_index("DateOnly")
