import streamlit as st
import pandas as pd
from data.api import get_news_data
from data.stockdata import get_stock_data
from data.preprocessing import process_sentiment_data, combine_data, prepare_sentiment_for_forecast
from models.forecasting import fit_and_forecast
from visualisation.plots import create_plot
from utils.helpers import calc_correlation, safe_utc_localize
from models.forecasting import fit_and_forecast
from visualisation.plots import create_plot, plot_price_with_forecast

st.sidebar.title("📊 Predicting Stock Prices by News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., MSFT):", value="MSFT")
forecast_days = st.sidebar.number_input("Forecast horizon (days):", min_value=3, max_value=30, value=7)
run_button = st.sidebar.button("Run Analysis")

if run_button:
    try:
        print("✅ Script started")

        # ---------- Fetch news & sentiment ----------
        news_data = get_news_data(ticker, days=30)
        if not news_data or "articles" not in news_data or "daily_sentiment" not in news_data:
            st.error("❌ get_news_data returned unexpected structure.")
            st.stop()

        articles_df = news_data["articles"]

        st.write("🔍 DEBUG: Article-level Sentiment Sample")
        if not articles_df.empty:
            show_cols = []
            for c in ["Date", "Title", "sentiment", "sentiment_score"]:
                if c in articles_df.columns:
                    show_cols.append(c)
            st.dataframe(articles_df[show_cols].head(10))
        else:
            st.write("No articles")

        # ---------- Process sentiment ----------
        result_df = process_sentiment_data(articles_df)

        # ✅ Ensure consistent datetime handling
        if "Date" in result_df.columns:
            result_df["Date"] = safe_utc_localize(result_df["Date"])
        if "DateOnly" in result_df.columns:
            result_df["DateOnly"] = pd.to_datetime(result_df["DateOnly"], errors="coerce").dt.floor("D")

        st.write("🔍 DEBUG: Processed Daily Sentiment")
        st.dataframe(result_df.head(15))

        if result_df is None or result_df.empty:
            st.error("❌ No processed sentiment available.")
            st.stop()

        start_date_val = result_df["DateOnly"].min()
        end_date_val = result_df["DateOnly"].max()

        if pd.isna(start_date_val) or pd.isna(end_date_val):
            st.error("❌ No valid dates found in news data.")
            st.stop()

        # ✅ Localize start/end dates for stock fetch
        start_date_val = safe_utc_localize(pd.Series([start_date_val]))[0]
        end_date_val = safe_utc_localize(pd.Series([end_date_val]))[0]

        # ---------- Fetch stock data ----------
        stock_data = get_stock_data(ticker, start_date_val, end_date_val)
        st.write("🔍 DEBUG: Stock Data Sample")
        st.dataframe(stock_data.head(10))

        # ---------- Merge stock & sentiment ----------
        combined_df = combine_data(result_df, stock_data)

        # ✅ Ensure datetime consistency after merge
        if "Date" in combined_df.columns:
            combined_df["Date"] = safe_utc_localize(combined_df["Date"])
        if "DateOnly" in combined_df.columns:
            combined_df["DateOnly"] = pd.to_datetime(combined_df["DateOnly"], errors="coerce").dt.floor("D")

        if combined_df is None or combined_df.empty:
            st.error("❌ Merged DataFrame is empty. Check stock/sentiment overlap.")
            st.stop()

        st.write("🔍 DEBUG: Combined Data Sample (check columns)")
        st.dataframe(combined_df.head(15))
        st.write("Columns:", combined_df.columns.tolist())

        # Quick sanity checks:
        if "Pct_Change" not in combined_df.columns:
            st.warning("⚠️ 'Pct_Change' not found — it will be computed from 'Close' if possible.")
            if "Close" in combined_df.columns:
                combined_df["Pct_Change"] = combined_df["Close"].pct_change() * 100.0

        # Check ranges
        st.write("🔍 Range check:")
        st.write("sentiment_score range:", combined_df["sentiment_score"].min(), combined_df["sentiment_score"].max())
        if "Pct_Change" in combined_df.columns:
            st.write("Pct_Change range:", combined_df["Pct_Change"].min(), combined_df["Pct_Change"].max())

        # ---------- Correlation ----------
        correlation = calc_correlation(stock_data, result_df)
        if correlation is None:
            st.warning("⚠️ Correlation could not be calculated.")
        else:
            st.write(f"📈 Correlation (lagged sentiment vs. % change): {correlation:.4f}")

        # ---------- Forecast ----------
        # Option B: direct sentiment-only forecasting
        sentiment_forecast_df = prepare_sentiment_for_forecast(result_df)
        forecast_result = fit_and_forecast(
            sentiment_forecast_df.rename(columns={"computed_score": "Pct_Change"}), 
            steps=forecast_days
        )
        

        # After combined_df is built
        results = fit_and_forecast(combined_df, steps=7)

        if results:
            st.subheader("📊 Forecast Results")
            st.write(f"**Recommendation:** {results['recommendation']} (certainty {results['certainty']:.2f})")

            st.write("### Daily Forecast")
            st.dataframe(results["daily_forecast"])

            st.write("### Weekly Forecast")
            st.dataframe(results["weekly_forecast"])

            # Plots
            create_plot(combined_df, forecast_df=results["daily_forecast"])
            plot_price_with_forecast(ticker, hist_days=30, forecast_df=results["daily_forecast"])
    
    except Exception as e:
    
        st.error(f"❌ Error during analysis: {e}")
