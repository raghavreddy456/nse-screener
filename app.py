import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="NSE Live Strategy & Intraday Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Active F&O and High-Beta Liquid Universe
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
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (30s)", value=False)
refresh_interval = st.sidebar.slider("Interval (seconds)", 15, 120, 30)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} IST")

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return round(100 - (100 / (1 + rs)), 1)

@st.cache_data(ttl=30)
def fetch_complete_snapshot():
    all_symbols = ACTIVE_NSE_UNIVERSE + list(SECTOR_MAP.values())
    raw = yf.download(all_symbols, period="3mo", interval="1d", group_by="ticker", progress=False)

    stock_rows = []
    for ticker in ACTIVE_NSE_UNIVERSE:
        try:
            if ticker not in raw.columns.levels[0]:
                continue
            df = raw[ticker].dropna(subset=["Close"])
            if len(df) < 22:
                continue

            open_p = float(df["Open"].iloc[-1])
            high_p = float(df["High"].iloc[-1])
            low_p = float(df["Low"].iloc[-1])
            ltp = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0

            if math.isnan(ltp) or prev_close == 0 or ltp <= 0:
                continue

            pct_change = round(((ltp - prev_close) / prev_close) * 100, 2)
            pivot = round((high_p + low_p + ltp) / 3, 2)
            day_range_pct = round(((high_p - low_p) / ltp) * 100, 2)
            vol_cr = round((vol * ltp) / 1e7, 2)

            avg_vol_20d = float(df["Volume"].iloc[-21:-1].mean()) if len(df) >= 21 else vol
            rvol = round(vol / avg_vol_20d, 2) if avg_vol_20d > 0 else 1.0
            rsi_val = calculate_rsi(df["Close"], 14)
            sma_50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else float(df["Close"].rolling(20).mean().iloc[-1])
            high_20d = float(df["High"].iloc[-21:-1].max()) if len(df) >= 21 else high_p

            # -----------------------------------------------------
            # Intraday Strategy Conditions
            # -----------------------------------------------------
            # Open = Low Bullish (tolerance within 0.15%)
            is_open_low = (abs(open_p - low_p) / open_p <= 0.0015) and (ltp > open_p) and (pct_change > 0.3)
            # Open = High Bearish (tolerance within 0.15%)
            is_open_high = (abs(high_p - open_p) / open_p <= 0.0015) and (ltp < open_p) and (pct_change < -0.3)
            # High Relative Volume Breakout
            is_vol_breakout = (rvol >= 1.3) and (abs(pct_change) >= 1.2)
            # Day Range Expansion (Volatile Momentum)
            is_range_exp = day_range_pct >= 2.5

            # -----------------------------------------------------
            # MTF Swing Conditions
            # -----------------------------------------------------
            is_50sma_pullback = (ltp >= sma_50 * 0.985) and (ltp <= sma_50 * 1.03) and (pct_change >= -0.5)
            is_20d_breakout = (high_p >= high_20d) and (rvol >= 1.25) and (pct_change > 0.8)
            is_rsi_pullback = (ltp > sma_50) and (40.0 <= rsi_val <= 53.0) and (pct_change > 0.0)

            stock_rows.append({
                "Symbol": ticker.replace(".NS", ""),
                "Open": round(open_p, 2),
                "Day High": round(high_p, 2),
                "Day Low": round(low_p, 2),
                "LTP": round(ltp, 2),
                "Change (%)": pct_change,
                "Day Range (%)": day_range_pct,
                "Pivot": pivot,
                "RVOL": rvol,
                "RSI (14)": rsi_val,
                "50 SMA": round(sma_50, 2),
                "Turnover (Cr)": vol_cr,
                "Flag_OpenLow": is_open_low,
                "Flag_OpenHigh": is_open_high,
                "Flag_VolBreakout": is_vol_breakout,
                "Flag_RangeExp": is_range_exp,
                "Flag_50SMA": is_50sma_pullback,
                "Flag_20DBreakout": is_20d_breakout,
                "Flag_RSIPullback": is_rsi_pullback
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

st.title("⚡ Dynamic NSE Strategy & Intraday Screener")
st.caption("Real-time relative strength scanner, institutional volume breakouts, and algorithmic execution setups.")

with st.spinner("Downloading real-time market quotes..."):
    df_stocks, df_sectors = fetch_complete_snapshot()

if df_stocks.empty:
    st.error("Market data currently unavailable. Please verify internet connectivity and refresh.")
    st.stop()

# Sector Performance Headers
best_sector = df_sectors.sort_values(by="Change (%)", ascending=False).iloc[0] if not df_sectors.empty else None
worst_sector = df_sectors.sort_values(by="Change (%)", ascending=True).iloc[0] if not df_sectors.empty else None

c1, c2, c3 = st.columns(3)
c1.metric("Universe Scanned", f"{len(df_stocks)} Equities")
if best_sector is not None:
    c2.metric("Leading Sector", str(best_sector["Sector"]), f"{best_sector['Change (%)']:+.2f}%")
if worst_sector is not None:
    c3.metric("Worst Drag Sector", str(worst_sector["Sector"]), f"{worst_sector['Change (%)']:+.2f}%", delta_color="inverse")

if worst_sector is not None:
    st.error(f"🚨 WEAKEST SECTOR: {worst_sector['Sector']} ({worst_sector['Change (%)']:+.2f}%) — Strictly avoid Long Intraday or MTF positions in this group.")

# Primary Tabs
tab_intra, tab_mtf, tab_volatility, tab_overview, tab_sectors = st.tabs([
    "🔥 Intraday Setups (Open=Low / RVOL)",
    "🎯 MTF Swing Strategy",
    "⚡ High Volatility & Big Movers",
    "📊 Market Gainers & Losers",
    "🌐 Sector Relative Strength"
])

# ---------------------------------------------------------
# Tab 1: Intraday Setups
# ---------------------------------------------------------
with tab_intra:
    st.subheader("🔥 Intraday Strategy Engine")
    
    intra_mode = st.radio(
        "Select Intraday Setup Condition:",
        [
            "🟢 Open = Low (Dominant Buyers from Opening Bell)",
            "🔴 Open = High (Aggressive Institutional Selling)",
            "🚀 High Relative Volume (RVOL >= 1.3x Breakout)",
            "💥 Range Expansion (Day Range >= 2.5% + Trend Above Pivot)"
        ],
        horizontal=True
    )

    if "Open = Low" in intra_mode:
        st.info("💡 **Condition:** Open matches Day Low (0.15% threshold) and price is trading above the opening tick with positive momentum. High probability long entries.")
        df_filtered = df_stocks[df_stocks["Flag_OpenLow"] & (df_stocks["LTP"] > df_stocks["Pivot"])].copy()
        df_filtered = df_filtered.sort_values(by="Change (%)", ascending=False)
        trade_bias = "BUY / LONG"

    elif "Open = High" in intra_mode:
        st.info("💡 **Condition:** Open matches Day High (0.15% threshold) and price is rejecting lower below the benchmark pivot. Ideal for intraday shorting.")
        df_filtered = df_stocks[df_stocks["Flag_OpenHigh"] & (df_stocks["LTP"] < df_stocks["Pivot"])].copy()
        df_filtered = df_filtered.sort_values(by="Change (%)", ascending=True)
        trade_bias = "SELL / SHORT"

    elif "High Relative Volume" in intra_mode:
        st.info("💡 **Condition:** Stocks running at $\ge1.3\times$ their 20-day average volume with strong percentage moves ($\ge1.2\%$).")
        df_filtered = df_stocks[df_stocks["Flag_VolBreakout"]].copy()
        df_filtered = df_filtered.sort_values(by="RVOL", ascending=False)
        trade_bias = "MOMENTUM"

    else:
        st.info("💡 **Condition:** Stocks expanding beyond 2.5% intraday range while holding above the day's benchmark pivot.")
        df_filtered = df_stocks[df_stocks["Flag_RangeExp"] & (df_stocks["LTP"] > df_stocks["Pivot"])].copy()
        df_filtered = df_filtered.sort_values(by="Day Range (%)", ascending=False)
        trade_bias = "EXPANSION BUY"

    if df_filtered.empty:
        st.warning("⚠️ No stocks currently qualify under the strict criteria for this strategy in this snapshot. Check during active market hours (9:30 AM – 3:00 PM).")
    else:
        display_cols = ["Symbol", "LTP", "Change (%)", "Open", "Day High", "Day Low", "RVOL", "Day Range (%)", "Pivot", "Turnover (Cr)"]
        st.dataframe(df_filtered[display_cols], use_container_width=True, hide_index=True)

        st.markdown(f"#### 🎯 Active Intraday Execution Cards ({trade_bias})")
        for _, row in df_filtered.head(5).iterrows():
            ltp = row["LTP"]
            is_long = row["Change (%)"] >= 0

            # Calculate precise stop-loss and targets based on ATR / Pivot risk
            if is_long:
                sl = min(row["Day Low"], round(ltp * 0.992, 2))
                risk = ltp - sl if (ltp - sl) > 0 else ltp * 0.008
                target_1 = round(ltp + (risk * 1.5), 2)
                target_2 = round(ltp + (risk * 2.0), 2)
            else:
                sl = max(row["Day High"], round(ltp * 1.008, 2))
                risk = sl - ltp if (sl - ltp) > 0 else ltp * 0.008
                target_1 = round(ltp - (risk * 1.5), 2)
                target_2 = round(ltp - (risk * 2.0), 2)

            with st.container(border=True):
                col_title, col_info = st.columns([3, 2])
                col_title.markdown(f"### **{row['Symbol']}** — ₹{ltp} ({row['Change (%)']:+.2f}%)")
                col_info.markdown(f"**RVOL:** `{row['RVOL']}x` | **Range:** `{row['Day Range (%)']}%` | **Pivot:** ₹{row['Pivot']}")

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Action", "BUY ON PULLBACK" if is_long else "SELL ON BOUNCE")
                e2.metric("Target 1 (1:1.5)", f"₹{target_1}")
                e3.metric("Target 2 (1:2.0)", f"₹{target_2}")
                e4.metric("Strict Stop Loss", f"₹{sl}", delta_color="inverse")

# ---------------------------------------------------------
# Tab 2: MTF Swing Strategy
# ---------------------------------------------------------
with tab_mtf:
    st.subheader("🎯 Positional MTF Swing Selector (4x Leverage)")
    mtf_strat = st.radio(
        "Select MTF Strategy:",
        [
            "1. 50-Day SMA Pullback (Trend-Following Dip Buy)",
            "2. 20-Day High Breakout + Volume (Aggressive Momentum)",
            "3. RSI 40-52 Support Reversal (Oversold Dip in Strong Stocks)"
        ],
        horizontal=True
    )

    if "50-Day SMA Pullback" in mtf_strat:
        df_mtf_filtered = df_stocks[df_stocks["Flag_50SMA"]].sort_values(by="Change (%)", ascending=False)
    elif "20-Day High Breakout" in mtf_strat:
        df_mtf_filtered = df_stocks[df_stocks["Flag_20DBreakout"]].sort_values(by="RVOL", ascending=False)
    else:
        df_mtf_filtered = df_stocks[df_stocks["Flag_RSIPullback"]].sort_values(by="RSI (14)", ascending=True)

    if df_mtf_filtered.empty:
        st.warning("⚠️ No stocks currently qualify under the strict swing criteria.")
    else:
        st.dataframe(
            df_mtf_filtered[["Symbol", "LTP", "Change (%)", "50 SMA", "RSI (14)", "RVOL", "Turnover (Cr)"]],
            use_container_width=True, hide_index=True
        )
        st.markdown("#### 🛒 Recommended MTF Orders")
        for _, row in df_mtf_filtered.head(5).iterrows():
            ltp = row["LTP"]
            margin = round(ltp * 0.22, 2)
            t1 = round(ltp * 1.055, 2)
            t2 = round(ltp * 1.10, 2)
            sl = round(ltp * 0.965, 2)
            with st.container(border=True):
                st.markdown(f"**{row['Symbol']}** — LTP: ₹{ltp} ({row['Change (%)']:+.2f}%) | Est. Upfront Margin: ~₹{margin}/sh")
                st.write(f"Target 1 (+5.5%): ₹{t1} | Target 2 (+10%): ₹{t2} | Stop Loss (-3.5%): ₹{sl} | 50 SMA: ₹{row['50 SMA']}")

# ---------------------------------------------------------
# Tab 3: High Volatility
# ---------------------------------------------------------
with tab_volatility:
    st.subheader("⚡ Top 10 High Volatility & Momentum Equities")
    st.caption("Ranked by Day Range Expansion & Turnover. Best for active intraday range riders.")
    df_vol = df_stocks.sort_values(by=["Day Range (%)", "Turnover (Cr)"], ascending=[False, False]).head(10)
    st.dataframe(
        df_vol[["Symbol", "LTP", "Change (%)", "Day Range (%)", "RSI (14)", "RVOL", "Turnover (Cr)"]],
        use_container_width=True, hide_index=True
    )

# ---------------------------------------------------------
# Tab 4: Overview
# ---------------------------------------------------------
with tab_overview:
    cg, cl = st.columns(2)
    with cg:
        st.subheader("🟢 Top 10 Gainers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=False).head(10)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)
    with cl:
        st.subheader("🔴 Top 10 Losers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=True).head(10)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Tab 5: Sectors
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