import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="NSE Live MTF & Algo Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Active MTF-eligible and high-liquidity stock universe
ACTIVE_NSE_UNIVERSE = [
    "ANGELONE.NS", "RBLBANK.NS", "CARTRADE.NS", "TATACHEM.NS", "SUZLON.NS",
    "BRIGADE.NS", "POLICYBZR.NS", "PAYTM.NS", "GODREJPROP.NS", "PFC.NS",
    "RECLTD.NS", "HUDCO.NS", "IRFC.NS", "RVNL.NS", "BHEL.NS", "HAL.NS",
    "BEL.NS", "DIXON.NS", "COFORGE.NS", "PERSISTENT.NS", "KPITTECH.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "VOLTAS.NS", "POLYCAB.NS",
    "TRENT.NS", "YESBANK.NS", "BANDHANBNK.NS", "NATIONALUM.NS", "RELIANCE.NS",
    "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "ONGC.NS",
    "KOTAKBANK.NS", "NTPC.NS", "AXISBANK.NS", "TITAN.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJ-AUTO.NS", "TECHM.NS",
    "EICHERMOT.NS", "M&M.NS", "COALINDIA.NS", "HINDALCO.NS", "APOLLOHOSP.NS",
    "WIPRO.NS", "NESTLEIND.NS", "GRASIM.NS", "JSWSTEEL.NS", "POWERGRID.NS",
    "DLF.NS", "JINDALSTEL.NS", "VEDL.NS", "TATASTEEL.NS"
]

SECTOR_MAP = {
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY REALTY": "^CNXREALTY"
}

st.sidebar.title("⚡ Screener Controls")
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (60s)", value=False)
refresh_interval = st.sidebar.slider("Interval (seconds)", 30, 180, 60)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} IST")

# ---------------------------------------------------------
# Technical Indicators & Data Pipeline
# ---------------------------------------------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return round(100 - (100 / (1 + rs)), 1)

@st.cache_data(ttl=45)
def fetch_multi_strategy_snapshot():
    all_symbols = ACTIVE_NSE_UNIVERSE + list(SECTOR_MAP.values())
    raw = yf.download(all_symbols, period="3mo", interval="1d", group_by="ticker", progress=False)

    stock_rows = []
    for ticker in ACTIVE_NSE_UNIVERSE:
        try:
            if ticker not in raw.columns.levels[0]:
                continue
            df = raw[ticker].dropna(subset=["Close"])
            if len(df) < 25:
                continue

            ltp = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            high = float(df["High"].iloc[-1])
            low = float(df["Low"].iloc[-1])
            vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0

            if math.isnan(ltp) or prev_close == 0:
                continue

            # Core Technicals
            pct_change = round(((ltp - prev_close) / prev_close) * 100, 2)
            pivot = round((high + low + ltp) / 3, 2)
            day_range_pct = round(((high - low) / ltp) * 100, 2)
            vol_cr = round((vol * ltp) / 1e7, 2)

            sma_20 = float(df["Close"].rolling(20).mean().iloc[-1])
            sma_50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else sma_20
            high_20d = float(df["High"].iloc[-21:-1].max()) if len(df) >= 21 else high
            avg_vol_20d = float(df["Volume"].iloc[-21:-1].mean()) if len(df) >= 21 else vol
            vol_surge = round(vol / avg_vol_20d, 2) if avg_vol_20d > 0 else 1.0
            rsi_val = calculate_rsi(df["Close"], 14)

            # Strategy Signal Flags
            # 1. 50-SMA Pullback: Price is above 50 SMA, but within 2.5% of it
            is_50sma_pullback = (ltp >= sma_50 * 0.985) and (ltp <= sma_50 * 1.03) and (pct_change >= -0.5)

            # 2. 20-Day Breakout: Today's high breaks prior 20-day high with volume > 1.3x
            is_breakout = (high >= high_20d) and (vol_surge >= 1.25) and (pct_change > 0.8)

            # 3. RSI Oversold Reversal: Bullish stock (above 50 SMA) with RSI between 40-52
            is_rsi_pullback = (ltp > sma_50) and (40.0 <= rsi_val <= 53.0) and (pct_change > 0.0)

            # 4. Momentum Above Pivot: Positive change, trading above daily pivot, strong volume
            is_momentum = (pct_change >= 1.5) and (ltp > pivot) and (vol_surge >= 1.2)

            stock_rows.append({
                "Symbol": ticker.replace(".NS", ""),
                "LTP": round(ltp, 2),
                "Change (%)": pct_change,
                "Day Range (%)": day_range_pct,
                "Pivot": pivot,
                "RSI (14)": rsi_val,
                "50 SMA": round(sma_50, 2),
                "20 SMA": round(sma_20, 2),
                "Vol Surge": vol_surge,
                "Turnover (Cr)": vol_cr,
                "Flag_50SMA": is_50sma_pullback,
                "Flag_Breakout": is_breakout,
                "Flag_RSI": is_rsi_pullback,
                "Flag_Momentum": is_momentum
            })
        except Exception:
            continue

    sector_rows = []
    for name, sym in SECTOR_MAP.items():
        try:
            if sym not in raw.columns.levels[0]:
                continue
            df_sec = raw[sym].dropna(subset=["Close"])
            if len(df_sec) < 2:
                continue
            ltp = float(df_sec["Close"].iloc[-1])
            prev = float(df_sec["Close"].iloc[-2])
            pct = round(((ltp - prev) / prev) * 100, 2)
            sector_rows.append({"Sector": name, "LTP": round(ltp, 2), "Change (%)": pct})
        except Exception:
            continue

    return pd.DataFrame(stock_rows), pd.DataFrame(sector_rows)

# ---------------------------------------------------------
# View Layer
# ---------------------------------------------------------
st.title("⚡ Dynamic NSE MTF Strategy Screener")
st.caption("Quantitative screener for high-momentum swings, 4x MTF leverage candidates, and sector divergence.")

with st.spinner("Analyzing technical structures & moving averages..."):
    df_stocks, df_sectors = fetch_multi_strategy_snapshot()

if df_stocks.empty:
    st.error("Market data currently unavailable. Please verify internet access and refresh.")
    st.stop()

# Sector Header
best_sector = df_sectors.sort_values(by="Change (%)", ascending=False).iloc[0] if not df_sectors.empty else None
worst_sector = df_sectors.sort_values(by="Change (%)", ascending=True).iloc[0] if not df_sectors.empty else None

c1, c2, c3 = st.columns(3)
c1.metric("Universe Scanned", f"{len(df_stocks)} Equities")
if best_sector is not None:
    c2.metric("Leading Sector", str(best_sector["Sector"]), f"{best_sector['Change (%)']:+.2f}%")
if worst_sector is not None:
    c3.metric("Worst Drag Sector", str(worst_sector["Sector"]), f"{worst_sector['Change (%)']:+.2f}%", delta_color="inverse")

if worst_sector is not None:
    st.error(f"🚨 WEAKEST SECTOR: {worst_sector['Sector']} ({worst_sector['Change (%)']:+.2f}%) — Avoid long MTF positions in this group.")

# Primary Strategy Tabs
tab_strategy_mtf, tab_volatility, tab_overview, tab_sectors = st.tabs([
    "🎯 MTF Strategy Selector",
    "⚡ High Volatility & Big Movers",
    "📊 Market Gainers & Losers",
    "🌐 Sector Relative Strength"
])

# ---------------------------------------------------------
# Tab 1: MTF Strategy Selector
# ---------------------------------------------------------
with tab_strategy_mtf:
    st.subheader("🎯 Select an MTF Buying Strategy")
    
    selected_strategy = st.radio(
        "Choose Quantitative Filter:",
        [
            "1. 50-Day SMA Pullback (Best Dip-Buy for 2-4 Week Swings)",
            "2. 20-Day High Breakout + Volume Surge (Aggressive Momentum)",
            "3. RSI 40-50 Reversal Bounce (Oversold Dip in Strong Stocks)",
            "4. VWAP Pivot Alignment (High Intraday Relative Strength)"
        ],
        horizontal=False
    )

    if "50-Day SMA Pullback" in selected_strategy:
        st.info("💡 **Logic:** Identifies stocks retesting their institutional 50-Day SMA (within $\pm2.5\%$). Provides tight stop-losses for 4x MTF positional leverage.")
        filtered_df = df_stocks[df_stocks["Flag_50SMA"]].copy()
        sort_col = "Change (%)"
        asc = False
    elif "20-Day High Breakout" in selected_strategy:
        st.info("💡 **Logic:** Stocks piercing their 20-day resistance on at least 1.25x volume expansion. Ideal for high-beta swing moves.")
        filtered_df = df_stocks[df_stocks["Flag_Breakout"]].copy()
        sort_col = "Vol Surge"
        asc = False
    elif "RSI 40-50 Reversal" in selected_strategy:
        st.info("💡 **Logic:** Stocks above 50 SMA with cooling RSI (40–53) showing early stabilization. High probability reversal entries.")
        filtered_df = df_stocks[df_stocks["Flag_RSI"]].copy()
        sort_col = "RSI (14)"
        asc = True
    else:
        st.info("💡 **Logic:** Clean intraday momentum (>1.5% gain) holding strictly above the benchmark pivot with high turnover.")
        filtered_df = df_stocks[df_stocks["Flag_Momentum"]].copy()
        sort_col = "Change (%)"
        asc = False

    if filtered_df.empty:
        st.warning("⚠️ No stocks currently qualify under the strict criteria for this strategy in this snapshot. Check the other strategy tabs.")
    else:
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=asc)

        # Detailed Table
        display_cols = ["Symbol", "LTP", "Change (%)", "50 SMA", "RSI (14)", "Vol Surge", "Pivot", "Turnover (Cr)"]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("#### 🛒 Top Recommended MTF Execution Orders")
        for _, row in filtered_df.head(5).iterrows():
            ltp = row["LTP"]
            est_margin = round(ltp * 0.22, 2)   # ~22% upfront requirement for 4.5x leverage
            target_1 = round(ltp * 1.055, 2)    # +5.5% Positional Target
            target_2 = round(ltp * 1.10, 2)     # +10% Extended Target
            sl = round(ltp * 0.965, 2)          # -3.5% Strict Swing SL

            with st.container(border=True):
                c_head, c_badge = st.columns([4, 1])
                c_head.markdown(f"### **{row['Symbol']}** — LTP: ₹{ltp} ({row['Change (%)']:+.2f}%)")
                c_badge.markdown(f"**RSI:** `{row['RSI (14)']}` | **Vol:** `{row['Vol Surge']}x`")

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Est. Upfront Margin", f"₹{est_margin}/sh", help="Assumes ~22% required cash margin under broker MTF")
                m_col2.metric("Target 1 (+5.5%)", f"₹{target_1}")
                m_col3.metric("Target 2 (+10.0%)", f"₹{target_2}")
                m_col4.metric("Stop Loss (-3.5%)", f"₹{sl}", delta_color="inverse")

# ---------------------------------------------------------
# Tab 2: High Volatility
# ---------------------------------------------------------
with tab_volatility:
    st.subheader("⚡ Top 10 High Volatility & Momentum Equities")
    st.caption("Stocks ranked by day range expansion (Day High - Day Low) & momentum. (e.g. AngelOne, RBL Bank, Midcap Tech)")
    df_vol = df_stocks.sort_values(by=["Day Range (%)", "Turnover (Cr)"], ascending=[False, False]).head(10)
    st.dataframe(
        df_vol[["Symbol", "LTP", "Change (%)", "Day Range (%)", "RSI (14)", "Vol Surge", "Turnover (Cr)"]],
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# Tab 3: Gainers & Losers
# ---------------------------------------------------------
with tab_overview:
    col_g, col_l = st.columns(2)
    with col_g:
        st.subheader("🟢 Top 10 Gainers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=False).head(10)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)
    with col_l:
        st.subheader("🔴 Top 10 Losers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=True).head(10)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Tab 4: Sectors
# ---------------------------------------------------------
with tab_sectors:
    st.subheader("🌐 Sector Relative Performance")
    if not df_sectors.empty:
        fig_sec = px.bar(
            df_sectors, x="Sector", y="Change (%)", color="Change (%)",
            color_continuous_scale=["#ef4444", "#30363d", "#10b981"],
            text="Change (%)", template="plotly_dark", height=340
        )
        st.plotly_chart(fig_sec, use_container_width=True)