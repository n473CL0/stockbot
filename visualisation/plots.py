# plots.py
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

def create_plot(merged_df, forecast_df=None):
    """Sentiment + stock % change + rolling correlation + forecast overlay."""
    fig = go.Figure()

    # Sentiment
    col = "computed_score" if "computed_score" in merged_df.columns else "sentiment_score"
    if col in merged_df.columns:
        sentiment_std = (merged_df[col] - merged_df[col].mean()) / (merged_df[col].std() + 1e-8)
        fig.add_trace(go.Scatter(
            x=merged_df.index,
            y=sentiment_std,
            name='Weighted Sentiment (std)',
            line=dict(color='blue'),
            mode='lines'
        ))

    # Stock % change
    if "Pct_Change" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df.index,
            y=merged_df["Pct_Change"],
            name="Stock % Change",
            line=dict(color='green'),
            yaxis="y2"
        ))

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Forecast"],
            name="Forecast (Pct_Change)",
            line=dict(color="red", dash="dash"),
            yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_df.index, forecast_df.index[::-1]]),
            y=np.concatenate([forecast_df["Lower"], forecast_df["Upper"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            yaxis="y2"
        ))

    # Rolling correlation
    if "Rolling_Corr" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df.index,
            y=merged_df["Rolling_Corr"],
            name="Rolling Correlation",
            line=dict(color='orange', dash="dot"),
            yaxis="y3"
        ))

    # Layout
    fig.update_layout(
        title="Sentiment vs Stock % Change & Correlation + Forecast",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Sentiment (std)", tickfont=dict(color="blue")),
        yaxis2=dict(title="Stock % Change", overlaying="y", side="right", tickfont=dict(color="green")),
        yaxis3=dict(title="Rolling Corr", overlaying="y", side="left", position=0.05, tickfont=dict(color="orange")),
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_price_with_forecast(ticker, hist_days=30, forecast_df=None):
    """Show last N days of stock price + overlay projected price path."""
    hist = yf.Ticker(ticker).history(period=f"{hist_days}d")
    if hist.empty:
        st.warning("No stock data available.")
        return

    fig = go.Figure()

    # Actual prices (candlestick)
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Price"
    ))

    if forecast_df is not None and not forecast_df.empty:
        # Convert % change forecast into projected price path
        last_close = hist["Close"].iloc[-1]
        projected = [last_close]
        for pct in forecast_df["Forecast"]:
            projected.append(projected[-1] * (1 + pct / 100.0))
        projected = projected[1:]  # drop initial

        # Overlay forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=projected,
            name="Projected Price",
            line=dict(color="red")
        ))

        # Confidence bands
        lower, upper = [], []
        price = last_close
        for low, up in zip(forecast_df["Lower"], forecast_df["Upper"]):
            price_low = price * (1 + low / 100.0)
            price_up = price * (1 + up / 100.0)
            lower.append(price_low)
            upper.append(price_up)
            price = price * (1 + forecast_df["Forecast"].iloc[0] / 100.0)  # approximate path

        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_df.index, forecast_df.index[::-1]]),
            y=np.concatenate([lower, upper[::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        title=f"{ticker} Price + Forecast (Last {hist_days} Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


def show_company_info_and_stock(ticker: str, days: int = 30):
    """Show company info and stock chart for the last N days."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = getattr(ticker_obj, "info", {}) or {}

        # --- Company Info ---
        st.subheader("🏢 Company Information")
        st.write(f"**Name:** {info.get('longName') or info.get('shortName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
        st.write(f"**P/E Ratio:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**52 Week Range:** {info.get('fiftyTwoWeekLow', 'N/A')} – {info.get('fiftyTwoWeekHigh', 'N/A')}")

        # --- Stock Chart ---
        st.subheader(f"📈 Stock Price (Last {days} days)")
        hist = ticker_obj.history(period=f"{days}d")
        if hist.empty:
            st.warning("No historical stock data available.")
            return

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Stock Price"
        )])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            title=f"{ticker} Stock Price - Last {days} Days"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Could not fetch company info/stock data: {e}")
