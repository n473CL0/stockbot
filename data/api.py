import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import feedparser
import streamlit as st
import time

from utils.helpers import normalize_sentiment_columns, compute_weighted_daily_sentiment, safe_utc_localize
from models.sentiment_llm import analyze_sentiment_finbert


def get_news_data(ticker: str, days: int = 30, decay: float = 0.95):
    """
    Fetch multi-source news, classify sentiment, and return:
    - articles (article-level DataFrame with sentiment scores)
    - daily_sentiment (aggregated daily scores with DateOnly & Date)
    """

    import traceback

    # === Debugging helpers ===
    def debug_wrap(label, func, *args, **kwargs):
        """Run a function with debugging wrapper."""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            st.error(f"❌ Error in {label}: {e}")
            st.code(traceback.format_exc())
            return None

    def debug_df(label, df):
        """Print debug info about DataFrame."""
        try:
            if df is None:
                st.warning(f"⚠️ {label} DataFrame is None")
                return
            if df.empty:
                st.warning(f"⚠️ {label} DataFrame is empty")
                return
            st.write(f"🔎 DEBUG [{label}] shape={df.shape}")
            st.write(df.head(5))
            if "Date" in df.columns:
                st.write(f"   {label} Date dtype:", df["Date"].dtype)
                st.write(f"   Sample dates:", df["Date"].head().tolist())
        except Exception as e:
            st.error(f"❌ Debug print failed for {label}: {e}")

    # === API keys & constants ===
    MARKET_AUX_KEY = "6ANHWMXsm2nqNjRUYPCycdXhCGMa4XxeO5htEf9k"
    TIINGO_KEY = "300a4febde0576bba242c112ad40224e892b5aba"
    NEWSAPI_KEY = "6dd722dc057a447d873c7908eda543ba"
    FINVIZ_URL = "https://finviz.com/quote.ashx?t="

    rows, counts_raw = [], {}
    st.write("🚀 Starting get_news_data()")
    t0 = time.time()

    # === Company metadata ===
    try:
        ticker_obj = yf.Ticker(ticker)
        info = getattr(ticker_obj, "info", {}) or {}
        company_name = info.get("longName") or info.get("shortName") or ticker
        entities = list({ticker, company_name})
    except Exception:
        company_name = ticker
        entities = [ticker]

    # Date window
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # (keep your is_finance_relevant + fetchers unchanged here...)
    finance_publishers = [
        "reuters","bloomberg","wsj","wall street journal","financial times","ft.com",
        "marketwatch","barron","cnbc","seeking alpha","tipranks","zacks","benzinga",
        "investing.com","yahoo finance","the motley fool","thestreet","morningstar",
        "investopedia","s&p global","fitch","moody","nasdaq","dow jones","citywire"
    ]
    include_kw = [
        "stock","share","shares","earnings","results","q1","q2","q3","q4",
        "guidance","outlook","forecast","analyst","upgrade","downgrade",
        "price target","pt","dividend","buyback","repurchase","revenue","profit",
        "net income","eps","ebitda","valuation","merger","acquisition","sec",
        "10-k","10q","10-q","8-k","premarket","after-hours","pre-market","after hours"
    ]
    exclude_kw = [
        "how to","review","recipe","fashion","lifestyle","travel","gaming tips",
        "video game review","celebrity","gossip","rumor","leak"
    ]

    def is_finance_relevant(title, publisher, link, src):
        t, p, l = (title or "").lower(), (publisher or "").lower(), (link or "").lower()
        text = f"{t} {p} {l}"
        if any(dom in text for dom in finance_publishers):
            pass_check = True
        else:
            pass_check = any(k in t for k in include_kw)
        if any(k in t for k in exclude_kw):
            return False
        if src == "YahooFinance":
            return True
        return pass_check

    # === Fetchers (all using safe_utc_localize) ===
    def fetch_finviz(ticker_sym):
        try:
            url = FINVIZ_URL + ticker_sym
            req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
            html = BeautifulSoup(urlopen(req), "html.parser")
            news_table = html.find(id="news-table")
            if news_table is None:
                return []
            parsed = []
            for row in news_table.find_all("tr"):
                a = row.find("a")
                if not a:
                    continue
                headline, link = a.get_text(strip=True), a.get("href", "")
                date_scrape = row.td.text.strip().split()
                if len(date_scrape) == 1:
                    if parsed:
                        date = parsed[-1]["Date"].date().strftime("%Y-%m-%d")
                    else:
                        date = datetime.utcnow().strftime("%Y-%m-%d")
                    time_str = date_scrape[0]
                else:
                    date, time_str = date_scrape[0], date_scrape[1]
                dt = safe_utc_localize(f"{date} {time_str}")
                if pd.isna(dt):
                    continue
                parsed.append(dict(
                    Title=headline, Link=link, Publisher="Finviz",
                    Date=dt, Source="Finviz", sentiment=None
                ))
            return parsed
        except Exception as e:
            st.warning(f"Finviz fetch failed: {e}")
            return []

    def fetch_marketaux(entity, start_date, end_date, api_key):
        try:
            url = (f"https://api.marketaux.com/v1/news/all?"
                   f"symbols={entity}&filter_entities=true&language=en&"
                   f"published_after={start_date}&published_before={end_date}&api_token={api_key}")
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return []
            data = r.json().get("data", [])
            return [
                dict(
                    Title=art.get("title",""), Link=art.get("url",""),
                    Publisher=art.get("source",""),
                    Date=safe_utc_localize(art.get("published_at")),
                    Source="Marketaux", sentiment=None
                )
                for art in data
            ]
        except Exception as e:
            st.warning(f"Marketaux fetch failed: {e}")
            return []

    def fetch_tiingo(entity, start_date, end_date, api_key):
        url = "https://api.tiingo.com/tiingo/news"
        headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}
        params = {"tickers": entity, "startDate": start_date, "endDate": end_date,
                  "limit": 1000, "sortBy": "publishedDate"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return [
                dict(
                    Title=d.get("title",""), Link=d.get("url",""),
                    Publisher=d.get("source","Tiingo"),
                    Date=safe_utc_localize(d.get("publishedDate")),
                    Source="Tiingo", sentiment=None
                )
                for d in data
            ]
        except Exception as e:
            st.warning(f"Tiingo fetch failed: {e}")
            return []

    def fetch_newsapi(entity, start_date, end_date, api_key):
        try:
            url = (f"https://newsapi.org/v2/everything?"
                   f"q={entity}&from={start_date}&to={end_date}&language=en&"
                   f"sortBy=publishedAt&apiKey={api_key}")
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return []
            data = r.json().get("articles", [])
            return [
                dict(
                    Title=art.get("title",""), Link=art.get("url",""),
                    Publisher=(art.get("source") or {}).get("name",""),
                    Date=safe_utc_localize(art.get("publishedAt")),
                    Source="NewsAPI", sentiment=None
                )
                for art in data
            ]
        except Exception as e:
            st.warning(f"NewsAPI fetch failed: {e}")
            return []

    def fetch_google_news(query, days=14, max_articles=120):
        try:
            q = f'"{query}" (stock OR shares OR earnings OR results OR guidance OR analyst OR finance OR dividend OR buyback)'
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            articles, cutoff = [], pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            for entry in feed.entries[:max_articles]:
                dt = safe_utc_localize(entry.get("published"))
                if pd.isna(dt) or dt < cutoff:
                    continue
                publisher = getattr(getattr(entry, "source", None), "title", "") or entry.get("source", "") or "Google News"
                articles.append(dict(
                    Title=entry.get("title", ""), Link=entry.get("link", ""),
                    Publisher=publisher, Date=dt,
                    Source="GoogleNews", sentiment=None
                ))
            return articles
        except Exception as e:
            st.warning(f"Google News fetch failed: {e}")
            return []

    def fetch_yahoo_finance(ticker_sym, days=14, max_articles=80):
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_sym}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            articles, cutoff = [], pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            for entry in feed.entries[:max_articles]:
                dt = safe_utc_localize(entry.get("published"))
                if pd.isna(dt) or dt < cutoff:
                    continue
                articles.append(dict(
                    Title=entry.get("title",""), Link=entry.get("link",""),
                    Publisher="Yahoo Finance", Date=dt,
                    Source="YahooFinance", sentiment=None
                ))
            return articles
        except Exception as e:
            st.warning(f"Yahoo Finance fetch failed: {e}")
            return []

    # === Parallel fetch ===
    with ThreadPoolExecutor() as ex:
        tasks = []
        for ent in entities:
            tasks += [
                ex.submit(fetch_marketaux, ent, start_date, end_date, MARKET_AUX_KEY),
                ex.submit(fetch_newsapi, ent, start_date, end_date, NEWSAPI_KEY),
                ex.submit(fetch_tiingo, ent, start_date, end_date, TIINGO_KEY),
                ex.submit(fetch_google_news, ent, 14)
            ]
        tasks += [
            ex.submit(fetch_yahoo_finance, ticker, 14),
            ex.submit(fetch_finviz, ticker)
        ]

        for f in as_completed(tasks):
            batch = f.result() or []
            rows.extend(batch)
            if batch:
                src = batch[0].get("Source", "Other")
                counts_raw[src] = counts_raw.get(src, 0) + len(batch)

    # === DataFrame build ===
    now_utc = pd.Timestamp.now(tz='UTC')
    for r in rows:
        if not r.get("Date") or pd.isna(r.get("Date")):
            r["Date"] = now_utc

    articles_df = pd.DataFrame(rows)
    debug_df("Articles (raw)", articles_df)

    if articles_df.empty:
        st.warning("⚠️ No news articles found.")
        return {"articles": pd.DataFrame(), "daily_sentiment": pd.DataFrame(columns=["DateOnly","Date","sentiment_score"])}

    # Normalize datetime
    articles_df["Date"] = safe_utc_localize(articles_df["Date"]).fillna(now_utc)
    articles_df["DateOnly"] = articles_df["Date"].dt.tz_convert(None).dt.floor("D")
    debug_df("Articles (normalized)", articles_df)

    # Dedupe
    articles_df = (articles_df.drop_duplicates(subset=["Link","Title"])
                               .sort_values("Date", ascending=False)
                               .reset_index(drop=True))

    # === Finance relevance filter ===
    def _keep(row):
        if row["Source"] == "GoogleNews":
            return is_finance_relevant(row["Title"], row["Publisher"], row["Link"], row["Source"])
        if row["Source"] == "YahooFinance":
            return True
        return True

    filtered_df = articles_df[articles_df.apply(_keep, axis=1)].copy()
    if filtered_df.empty:
        filtered_df = articles_df.copy()
    debug_df("Articles (filtered)", filtered_df)

    if "sentiment" not in filtered_df.columns:
        filtered_df["sentiment"] = np.nan

    mask = filtered_df["sentiment"].isna()
    if mask.any():
        st.info(f"🧠 Running sentiment classification on {mask.sum()} articles...")
        try:
            titles = filtered_df.loc[mask, "Title"].astype(str).tolist()
            preds = analyze_sentiment_finbert(titles)

            def _normalize_pred(p):
                try:
                    if p is None: return 0.0
                    if isinstance(p, dict):
                        label, score = str(p.get("label","")).lower(), p.get("score") or p.get("sentiment") or p.get("value") or p.get("prob")
                        score = float(score) if score is not None else None
                        if score is None:
                            if "pos" in label: return 1.0
                            if "neg" in label: return -1.0
                            return 0.0
                        return -abs(score) if "neg" in label else abs(score) if "pos" in label else float(score)
                    if isinstance(p, (int,float,np.floating,np.integer)):
                        return float(p)
                    s = str(p).strip().lower()
                    if s in ("positive","pos","+","p"): return 1.0
                    if s in ("negative","neg","-","n"): return -1.0
                    if s in ("neutral","neu","0"): return 0.0
                    if ":" in s:
                        lab,val = s.split(":",1)
                        try: return float(val) if "pos" in lab else -float(val) if "neg" in lab else float(val)
                        except: return 0.0
                    return float(s)
                except: return 0.0

            preds = preds if isinstance(preds, list) else [preds]
            normed = [_normalize_pred(p) for p in preds]

            if len(normed) == mask.sum():
                filtered_df.loc[mask,"sentiment"] = normed
            else:
                L = min(len(normed), mask.sum())
                idx = filtered_df.loc[mask].index.tolist()
                for i in range(L):
                    filtered_df.at[idx[i], "sentiment"] = normed[i]
        finally:
            st.success("✅ Sentiment classification done")

    filtered_df["sentiment_score"] = pd.to_numeric(filtered_df["sentiment"], errors="coerce").fillna(0.0)

    print("🔍 DEBUG: Articles", filtered_df[["Title","Date","sentiment_score"]].head(10))

    # === Sentiment classification ===
    if "sentiment" not in filtered_df.columns:
        filtered_df["sentiment"] = np.nan

    mask = filtered_df["sentiment"].isna()
    if mask.any():
        st.info(f"🧠 Running sentiment classification on {mask.sum()} articles...")
        try:
            titles = filtered_df.loc[mask, "Title"].astype(str).tolist()
            preds = analyze_sentiment_finbert(titles)

            # (keep your _normalize_pred logic here...)
            preds = preds if isinstance(preds, list) else [preds]
            normed = [_normalize_pred(p) for p in preds]

            if len(normed) == mask.sum():
                filtered_df.loc[mask,"sentiment"] = normed
            else:
                L = min(len(normed), mask.sum())
                idx = filtered_df.loc[mask].index.tolist()
                for i in range(L):
                    filtered_df.at[idx[i], "sentiment"] = normed[i]
        finally:
            st.success("✅ Sentiment classification done")

    filtered_df["sentiment_score"] = pd.to_numeric(filtered_df["sentiment"], errors="coerce").fillna(0.0)
    debug_df("Articles (with sentiment)", filtered_df)

    # === Daily sentiment aggregation ===
    filtered_df = debug_wrap("normalize_sentiment_columns", normalize_sentiment_columns, filtered_df)
    debug_df("Filtered (post normalize)", filtered_df)

    daily_sentiment = debug_wrap("compute_weighted_daily_sentiment", compute_weighted_daily_sentiment, filtered_df, decay_val=decay)
    debug_df("Daily sentiment (raw)", daily_sentiment)

    if daily_sentiment is not None:
        daily_sentiment["DateOnly"] = pd.to_datetime(daily_sentiment["DateOnly"], errors="coerce").dt.normalize()
        daily_sentiment["sentiment_score"] = pd.to_numeric(daily_sentiment["sentiment_score"], errors="coerce").fillna(0.0)
        daily_sentiment["Date"] = safe_utc_localize(daily_sentiment["DateOnly"])
        debug_df("Daily sentiment (final)", daily_sentiment)

        # === Padding ===
        min_days = 15
        end_day = pd.Timestamp.now().normalize()
        full_range = pd.date_range(end=end_day, periods=min_days, freq="D")
        missing = full_range.difference(daily_sentiment["DateOnly"])
        if not missing.empty:
            pad = pd.DataFrame({
                "DateOnly": missing,
                "sentiment_score": 0.0,
                "Date": safe_utc_localize(missing)
            })
            daily_sentiment = pd.concat([daily_sentiment,pad],ignore_index=True).sort_values("DateOnly").reset_index(drop=True)

    else:
        daily_sentiment = pd.DataFrame(columns=["DateOnly","Date","sentiment_score"])

    # === Report ===
    zero_days = daily_sentiment.loc[daily_sentiment["sentiment_score"].abs()<1e-8,"DateOnly"].dt.strftime("%Y-%m-%d").tolist()
    if zero_days:
        st.info(f"ℹ️ Zero sentiment days: {', '.join(zero_days)}")

    st.info(f"📰 Total raw articles: {len(articles_df)}")
    st.write("   • Per-source (raw):", {k:int(v) for k,v in counts_raw.items()})
    st.info(f"🧹 Kept after filter: {len(filtered_df)}")
    st.write("   • Per-source (kept):", filtered_df["Source"].value_counts().to_dict())
    st.info(f"📅 Sentiment coverage days: {len(daily_sentiment)}")
    st.success(f"⏱️ Finished in {time.time()-t0:.2f}s")

    return {"articles": filtered_df, "daily_sentiment": daily_sentiment}
