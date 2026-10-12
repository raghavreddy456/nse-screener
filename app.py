import time
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="NSE Live Strategy Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "ONGC.NS", "KOTAKBANK.NS", "NTPC.NS", "AXISBANK.NS", "TITAN.NS",
    "ADANIENT.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJ-AUTO.NS", "TECHM.NS",
    "EICHERMOT.NS", "M&M.NS", "COALINDIA.NS", "HINDALCO.NS", "APOLLOHOSP.NS",
    "WIPRO.NS", "NESTLEIND.NS", "GRASIM.NS", "JSWSTEEL.NS", "POWERGRID.NS"
]

SECTOR_MAP = {
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY INFRA": "^CNXINFRA"
}

st.sidebar.title("⚡ Screener Controls")
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (30s)", value=False)
refresh_interval = st.sidebar.slider("Interval (seconds)", 15, 120, 30)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} IST")

@st.cache_data(ttl=30)
def fetch_market_snapshot():
    all_symbols = NIFTY_50_TICKERS + list(SECTOR_MAP.values())
    raw = yf.download(all_symbols, period="5d", interval="1d", group_by="ticker", progress=False)

    stock_rows = []
    for ticker in NIFTY_50_TICKERS:
        try:
            if ticker not in raw.columns.levels[0]:
                continue
            sub = raw[ticker].dropna(subset=["Close"])
            if len(sub) == 0:
                continue

            ltp = float(sub["Close"].iloc[-1])
            prev_close = float(sub["Close"].iloc[-2]) if len(sub) >= 2 else float(sub["Open"].iloc[-1])
            high = float(sub["High"].iloc[-1])
            low = float(sub["Low"].iloc[-1])
            vol = float(sub["Volume"].iloc[-1]) if "Volume" in sub.columns else 0.0

            if math.isnan(ltp) or math.isnan(prev_close) or prev_close == 0:
                continue

            pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            pivot = round((high + low + ltp) / 3, 2) if not math.isnan(high) else ltp
            vol_cr = round((vol * ltp) / 1e7, 2) if not math.isnan(vol) else 0.0

            clean_sym = ticker.replace(".NS", "")
            stock_rows.append({
                "Symbol": clean_sym,
                "LTP": round(ltp, 2),
                "Change (%)": pct,
                "Pivot": pivot,
                "Turnover (Cr)": vol_cr
            })
        except Exception:
            continue

    sector_rows = []
    for name, sym in SECTOR_MAP.items():
        try:
            if sym not in raw.columns.levels[0]:
                continue
            sub = raw[sym].dropna(subset=["Close"])
            if len(sub) == 0:
                continue

            ltp = float(sub["Close"].iloc[-1])
            prev_close = float(sub["Close"].iloc[-2]) if len(sub) >= 2 else float(sub["Open"].iloc[-1])

            if math.isnan(ltp) or math.isnan(prev_close) or prev_close == 0:
                continue

            pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            sector_rows.append({
                "Sector": name,
                "LTP": round(ltp, 2),
                "Change (%)": pct
            })
        except Exception:
            continue

    df_stk = pd.DataFrame(stock_rows)
    df_sec = pd.DataFrame(sector_rows)
    
    if not df_stk.empty:
        df_stk = df_stk.dropna(subset=["LTP", "Change (%)", "Pivot"])
    if not df_sec.empty:
        df_sec = df_sec.dropna(subset=["LTP", "Change (%)"])
        
    return df_stk, df_sec

st.title("⚡ Dynamic NSE Strategy Screener")
st.caption("Live market screening for Top Gainers/Losers, Sector Relative Strength, and Strategy Baskets.")

with st.spinner("Downloading real-time NSE market data via Yahoo Finance..."):
    df_stocks, df_sectors = fetch_market_snapshot()

if df_stocks.empty:
    st.error("Market data currently unavailable. Please verify internet access and refresh.")
    st.stop()

df_gainers = df_stocks.sort_values(by="Change (%)", ascending=False).head(10)
df_losers = df_stocks.sort_values(by="Change (%)", ascending=True).head(10)

best_sector = df_sectors.sort_values(by="Change (%)", ascending=False).iloc[0] if not df_sectors.empty else None
worst_sector = df_sectors.sort_values(by="Change (%)", ascending=True).iloc[0] if not df_sectors.empty else None

m1, m2, m3 = st.columns(3)
m1.metric("Universe Scanned", f"{len(df_stocks)} Stocks")
if best_sector is not None:
    m2.metric("Leading Sector", str(best_sector["Sector"]), f"{best_sector['Change (%)']:+.2f}%")
if worst_sector is not None:
    m3.metric("Worst Drag Sector", str(worst_sector["Sector"]), f"{worst_sector['Change (%)']:+.2f}%", delta_color="inverse")

if worst_sector is not None:
    st.error(f"🚨 HEAVIEST SECTOR DRAG: {worst_sector['Sector']} ({worst_sector['Change (%)']:+.2f}%) — Invalidate bullish setups in this sector's stocks.")

df_stocks["Abs_Momentum"] = df_stocks["Change (%)"].abs()
df_options = df_stocks.sort_values(by="Abs_Momentum", ascending=False).head(5)
df_mtf = df_stocks[(df_stocks["Change (%)"] > 0.1) & (df_stocks["LTP"] > df_stocks["Pivot"])].sort_values(by="Change (%)", ascending=False).head(5)

tab_movers, tab_options_ui, tab_mtf_ui = st.tabs([
    "📊 Live Gainers, Losers & Sectors",
    "🎯 Dynamic 5 Option Trades",
    "📈 Dynamic 5 MTF Swing Trades"
])

with tab_movers:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟢 Top 10 Gainers")
        st.dataframe(df_gainers[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("🔴 Top 10 Losers")
        st.dataframe(df_losers[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🌐 Sector Relative Performance")
    if not df_sectors.empty:
        fig_sec = px.bar(
            df_sectors, x="Sector", y="Change (%)", color="Change (%)",
            color_continuous_scale=["#ef4444", "#30363d", "#10b981"],
            text="Change (%)", template="plotly_dark", height=320
        )
        st.plotly_chart(fig_sec, use_container_width=True)

with tab_options_ui:
    st.subheader("🎯 5 Dynamically Screened Option Setups")
    for _, row in df_options.iterrows():
        ltp = float(row["LTP"])
        if math.isnan(ltp) or ltp <= 0:
            continue
            
        is_bull = row["Change (%)"] > 0
        bias = "BULLISH (Call Spread / Long CE)" if is_bull else "BEARISH (Put Spread / Long PE)"
        
        step = 50 if ltp > 1000 else (10 if ltp > 200 else 5)
        strike = int(round(ltp / step) * step)
        target = round(ltp * (1.025 if is_bull else 0.975), 2)
        sl = round(row["Pivot"] if is_bull else row["Pivot"] * 1.015, 2)

        with st.container(border=True):
            st.markdown(f"**{row['Symbol']}** — LTP: ₹{ltp} ({row['Change (%)']:+.2f}%) | **Bias: {bias}**")
            st.write(f"Strike: `{strike} {'CE' if is_bull else 'PE'}` | Intraday Pivot: ₹{row['Pivot']} | Target: ₹{target} | SL: ₹{sl}")

with tab_mtf_ui:
    st.subheader("📈 5 Dynamically Screened MTF Positional Swings")
    if df_mtf.empty:
        st.info("No stocks currently meet MTF criteria (positive change + price above pivot).")
    else:
        for _, row in df_mtf.iterrows():
            ltp = float(row["LTP"])
            if math.isnan(ltp) or ltp <= 0:
                continue
                
            margin = round(ltp * 0.22, 2)
            target = round(ltp * 1.06, 2)
            sl = round(ltp * 0.965, 2)

            with st.container(border=True):
                st.markdown(f"**{row['Symbol']}** — LTP: ₹{row['LTP']} ({row['Change (%)']:+.2f}%)")
                st.write(f"Est. Margin: ~₹{margin}/share | Support Pivot: ₹{row['Pivot']} | Target (+6%): ₹{target} | SL (-3.5%): ₹{sl}")