# helpers.py
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------
# Utilities
# -------------------------
def _to_numeric_sentiment(df):
    """Ensure a numeric 'sentiment_score' column exists."""
    if "sentiment_score" in df.columns:
        return pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

    if "sentiment" in df.columns:
        s = df["sentiment"]
        if np.issubdtype(s.dtype, np.number):
            return pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (
            s.astype(str).str.upper().map(
                {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
            ).fillna(0.0)
        )

    if "label" in df.columns and "score" in df.columns:
        labels = df["label"].astype(str).str.upper()
        scores = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        mapped = labels.map({"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0})
        return (mapped * scores).fillna(0.0)

    if "compound" in df.columns:
        return pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)

    return pd.Series(0.0, index=df.index, name="sentiment_score")


def _ensure_datetime_col(df, col_name):
    if col_name in df.columns:
        return pd.to_datetime(df[col_name], errors="coerce", utc=True)
    return None


import pandas as pd


def normalize_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we always have a numeric sentiment_score column.
    - Trusts sentiment_score from api.py
    - Falls back to 0.0 if missing or invalid
    """
    df = df.copy()

    if "sentiment_score" not in df.columns:
        # If something went wrong upstream
        df["sentiment_score"] = 0.0

    df["sentiment_score"] = (
        pd.to_numeric(df["sentiment_score"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    return df

# -------------------------
# Daily Sentiment
# -------------------------
def compute_weighted_daily_sentiment(
    df: pd.DataFrame,
    decay_val: float = 0.95,
    min_days: int = 15,
    debug: bool = False,
    pad_missing: bool = False,   # NEW flag
) -> pd.DataFrame:
    """
    Aggregate article-level sentiment into daily sentiment.
    Uses `computed_score` (weighted average) as canonical signal.
    By default returns only actual computed rows (no padding).
    Set pad_missing=True if you need a fixed-length display.
    """
    if df is None or df.empty:
        if pad_missing:
            end_day = pd.Timestamp.now().normalize()
            full_range = pd.date_range(end=end_day, periods=min_days, freq="D")
            out = pd.DataFrame({"DateOnly": full_range, "computed_score": 0.0})
            out["Date"] = safe_utc_localize(out["DateOnly"])
            out["sentiment_score"] = out["computed_score"]
            return out
        return pd.DataFrame(columns=["DateOnly", "computed_score", "sentiment_score", "Date"])

    df = normalize_sentiment_columns(df)

    # --- Normalize dates ---
    if "Date" in df.columns:
        df["Date"] = safe_utc_localize(df["Date"])
        df = df.dropna(subset=["Date"])
        # keep DateOnly tz-naive
        df["DateOnly"] = df["Date"].dt.tz_convert(None).dt.normalize()
    elif "DateOnly" in df.columns:
        df["DateOnly"] = pd.to_datetime(df["DateOnly"], errors="coerce")
        df = df.dropna(subset=["DateOnly"])
    else:
        raise KeyError("Input must have a 'Date' or 'DateOnly' column")

    out = []
    group_debug = []
    for i, (day, group) in enumerate(df.groupby("DateOnly")):
        group = group.sort_values(
            "Date" if "Date" in group.columns else "DateOnly",
            ascending=False
        ).reset_index(drop=True)

        nonzero = group[group["sentiment_score"].abs() > 1e-8]
        used = nonzero if not nonzero.empty else group

        weights = np.power(decay_val, np.arange(len(used)))
        w = weights / weights.sum() if weights.sum() > 0 else np.ones(len(used)) / len(used)

        try:
            score = float(np.average(used["sentiment_score"].astype(float), weights=w))
        except Exception:
            score = 0.0

        out.append({
            "DateOnly": pd.Timestamp(day).normalize(),
            "computed_score": float(np.clip(score, -1.0, 1.0))
        })

        if debug and i < 30:
            group_debug.append({
                "DateOnly": str(pd.Timestamp(day).date()),
                "group_size": int(len(group)),
                "nonzero_count": int(len(nonzero)),
                "used_values": [round(x, 6) for x in used["sentiment_score"].tolist()],
                "weights": [round(x, 6) for x in w.tolist()],
                "computed_score": float(np.clip(score, -1.0, 1.0))
            })

    daily = pd.DataFrame(out).sort_values("DateOnly").reset_index(drop=True)

    # --- Add back Date (always UTC tz-aware) + compat column ---
    daily["Date"] = safe_utc_localize(daily["DateOnly"])
    daily["sentiment_score"] = daily["computed_score"]

    # --- Optional padding ---
    if pad_missing:
        end_day = pd.Timestamp.now().normalize()
        full_range = pd.date_range(end=end_day, periods=min_days, freq="D")
        daily = (
            daily.set_index("DateOnly")
            .reindex(full_range)
            .rename_axis("DateOnly")
            .reset_index()
        )
        daily["computed_score"] = daily["computed_score"].fillna(0.0)
        daily["sentiment_score"] = daily["computed_score"]
        daily["Date"] = safe_utc_localize(daily["DateOnly"])

    if debug:
        st.write("🔍 Per-day debug (first groups)")
        st.dataframe(pd.DataFrame(group_debug))
        zero_days = daily.loc[daily["computed_score"].abs() < 1e-8, "DateOnly"].dt.strftime("%Y-%m-%d").tolist()
        st.write(f"🔍 Days with 0 after aggregation: {zero_days}")

    return daily





def ensure_daily_sentiment(sentiment_df: pd.DataFrame, decay: float = 0.95) -> pd.DataFrame:
    """
    Always compute weighted daily sentiment from article- or daily-level input.
    Accepts either 'Date' or 'DateOnly'.
    """
    if sentiment_df is None or sentiment_df.empty:
        return pd.DataFrame(columns=["DateOnly", "sentiment_score", "Date"])

    df = sentiment_df.copy()
    return compute_weighted_daily_sentiment(df, decay_val=decay)




# -------------------------
# Correlation calculations
# -------------------------
def calc_correlation(stock_df, sentiment_df, decay=0.95):
    try:
        if stock_df is None or sentiment_df is None:
            st.error("❌ Missing stock or sentiment data for correlation.")
            return None

        stock = stock_df.copy()
        if "DateOnly" not in stock.columns:
            if "Date" in stock.columns:
                stock["Date"] = _ensure_datetime_col(stock, "Date")
                stock["DateOnly"] = stock["Date"].dt.normalize()
            else:
                st.error("❌ Stock data must have 'Date' or 'DateOnly'.")
                return None

        stock["DateOnly"] = pd.to_datetime(stock["DateOnly"], errors="coerce").dt.tz_localize(None)
        stock.dropna(subset=["DateOnly"], inplace=True)

        daily_sent = ensure_daily_sentiment(sentiment_df, decay=decay)
        if daily_sent is None or daily_sent.empty:
            st.warning("⚠️ Daily sentiment is empty — cannot calculate correlation.")
            return None

        daily_sent = daily_sent.copy()
        daily_sent["DateOnly"] = pd.to_datetime(daily_sent["DateOnly"], errors="coerce").dt.tz_localize(None)
        daily_sent.dropna(subset=["DateOnly"], inplace=True)

        merged = pd.merge(
            stock[["DateOnly", "Pct_Change"]],
            daily_sent[["DateOnly", "sentiment_score"]],
            on="DateOnly",
            how="inner"
        )

        if merged.empty:
            st.warning("⚠️ No overlapping dates for correlation.")
            return None

        merged["Lagged_Sentiment"] = merged["sentiment_score"].shift(1)
        corr = merged["Lagged_Sentiment"].corr(merged["Pct_Change"])
        return None if pd.isna(corr) else float(corr)

    except Exception as e:
        st.error(f"❌ Correlation calc failed: {e}")
        return None


def compute_rolling_correlation(stock_df, sentiment_df, window=30, decay=0.95):
    """Compute rolling correlation between lagged sentiment and stock % change."""
    daily_sent = ensure_daily_sentiment(sentiment_df, decay=decay)
    if daily_sent is None or daily_sent.empty:
        st.warning("⚠️ Daily sentiment is empty — cannot compute rolling correlation.")
        return None

    daily_sent = daily_sent.copy()
    daily_sent["DateOnly"] = pd.to_datetime(daily_sent["DateOnly"], errors="coerce").dt.tz_localize(None)

    stock = stock_df.copy()
    if "DateOnly" not in stock.columns:
        if "Date" in stock.columns:
            stock["Date"] = _ensure_datetime_col(stock, "Date")
            stock["DateOnly"] = stock["Date"].dt.normalize()
        else:
            return None

    stock["DateOnly"] = pd.to_datetime(stock["DateOnly"], errors="coerce").dt.tz_localize(None)
    stock.dropna(subset=["DateOnly"], inplace=True)

    merged = pd.merge(
        stock[["DateOnly", "Pct_Change"]],
        daily_sent[["DateOnly", "sentiment_score"]],
        on="DateOnly",
        how="inner"
    )
    if merged.empty:
        st.warning("⚠️ No overlapping dates for rolling correlation.")
        return None

    merged["Lagged_Sentiment"] = merged["sentiment_score"].shift(1)
    rolling_corr = (
        merged[["Lagged_Sentiment", "Pct_Change"]]
        .rolling(window=window, min_periods=max(5, window // 3))
        .corr()
        .unstack()
        .iloc[:, 1]
    )
    merged = merged.set_index("DateOnly")
    merged["Rolling_Corr"] = rolling_corr
    return merged


# -------------------------
# Safe UI helper
# -------------------------
def format_corr(corr_value):
    """Format correlation safely for display."""
    return f"{corr_value:.4f}" if corr_value is not None and not pd.isna(corr_value) else "n/a"

import pandas as pd
import streamlit as st

# utils/helpers.py
import pandas as pd
import streamlit as st

def safe_utc_localize(values) -> pd.Series | pd.Timestamp:
    """
    Ensure datetime values are tz-aware UTC.
    - Handles Series, DatetimeIndex, list-like, scalar, None
    - If naive -> localize to UTC
    - If tz-aware -> convert to UTC
    - Returns same shape as input
    """
    if values is None:
        st.write("⚠️ [safe_utc_localize] Input is None → returning empty UTC Series")
        return pd.Series(dtype="datetime64[ns, UTC]")

    # Normalize: if list-like but not Series/DatetimeIndex, convert to Series
    if isinstance(values, (list, tuple)):
        values = pd.Series(values)

    vals = pd.to_datetime(values, errors="coerce")

    # --- Debug logging ---
    try:
        if isinstance(vals, pd.Series):
            st.write("🛠️ [safe_utc_localize] Input Series", {
                "dtype": str(vals.dtype),
                "tz": str(getattr(vals.dt, "tz", None)),
                "na_count": int(vals.isna().sum()),
                "sample": str(vals.dropna().iloc[0]) if not vals.dropna().empty else "empty"
            })
        elif isinstance(vals, pd.DatetimeIndex):
            st.write("🛠️ [safe_utc_localize] Input DatetimeIndex", {
                "dtype": str(vals.dtype),
                "tz": str(vals.tz),
                "na_count": int(vals.isna().sum()),
                "sample": str(vals[0]) if len(vals) > 0 else "empty"
            })
        else:  # scalar
            st.write("🛠️ [safe_utc_localize] Input Scalar", {
                "type": str(type(vals)),
                "tz": getattr(vals, "tz", None),
                "value": str(vals)
            })
    except Exception as e:
        st.warning(f"⚠️ [safe_utc_localize debug failed]: {e}")

    # --- Handle Series ---
    if isinstance(vals, pd.Series):
        if vals.dt.tz is None:
            st.write("🔧 [safe_utc_localize] Localizing Series -> UTC")
            return vals.dt.tz_localize("UTC")
        else:
            st.write("🔧 [safe_utc_localize] Converting Series -> UTC")
            return vals.dt.tz_convert("UTC")

    # --- Handle DatetimeIndex ---
    if isinstance(vals, pd.DatetimeIndex):
        if vals.tz is None:
            st.write("🔧 [safe_utc_localize] Localizing Index -> UTC")
            return vals.tz_localize("UTC")
        else:
            st.write("🔧 [safe_utc_localize] Converting Index -> UTC")
            return vals.tz_convert("UTC")

    # --- Handle scalar (Timestamp or NaT) ---
    if pd.isna(vals):
        return vals
    if getattr(vals, "tz", None) is None:
        st.write("🔧 [safe_utc_localize] Localizing Scalar -> UTC")
        return vals.tz_localize("UTC")
    else:
        st.write("🔧 [safe_utc_localize] Converting Scalar -> UTC")
        return vals.tz_convert("UTC")




