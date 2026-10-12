import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="NSE Live Multi-Strategy Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 150+ Highly Liquid NSE F&O & High-Beta Midcap Universe
EXPANDED_NSE_UNIVERSE = [
    # High Beta / Volatile Swing & Momentum
    "ANGELONE.NS", "RBLBANK.NS", "CARTRADE.NS", "TATACHEM.NS", "SUZLON.NS",
    "BRIGADE.NS", "POLICYBZR.NS", "PAYTM.NS", "GODREJPROP.NS", "PFC.NS",
    "RECLTD.NS", "HUDCO.NS", "IRFC.NS", "RVNL.NS", "BHEL.NS", "HAL.NS",
    "BEL.NS", "DIXON.NS", "COFORGE.NS", "PERSISTENT.NS", "KPITTECH.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "VOLTAS.NS", "POLYCAB.NS",
    "TRENT.NS", "YESBANK.NS", "BANDHANBNK.NS", "NATIONALUM.NS", "IDEA.NS",
    "ZOMATO.NS", "TATAPOWER.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS",
    "GMRINFRA.NS", "ASHOKLEY.NS", "EXIDEIND.NS", "AMBUJACEM.NS", "ACC.NS",
    "CANBK.NS", "BANKBARODA.NS", "PNB.NS", "UNIONBANK.NS", "INDHOTEL.NS",
    "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS", "DLF.NS", "LODHA.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "M&MFIN.NS", "LICHSGFIN.NS",
    "BAJAJHLDNG.NS", "SHRIRAMFIN.NS", "POONAWALLA.NS", "MAXHEALTH.NS", "FORTIS.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "GLENMARK.NS", "TORNTPHARM.NS",
    "ALKEM.NS", "IPCALAB.NS", "LAURUSLABS.NS", "SYNGENE.NS", "DIVISLAB.NS",
    "DEEPAKNTR.NS", "NAVINFLUOR.NS", "SRF.NS", "PIIND.NS", "ASTRAL.NS",
    "SUPREMEIND.NS", "HAVELLS.NS", "CROMPTON.NS", "KEI.NS", "APLAPOLLO.NS",
    "BHARATFORG.NS", "BALKRISIND.NS", "MRF.NS", "APOLLOTYRE.NS", "MOTHERSON.NS",
    "BOSCHLTD.NS", "TIINDIA.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "ESCORTS.NS",
    "CUMMINSIND.NS", "ABB.NS", "SIEMENS.NS", "THERMAX.NS", "CGPOWER.NS",
    "KAYNES.NS", "CYIENT.NS", "MPHASIS.NS", "LTIM.NS", "LTTS.NS",
    "TATAELXSI.NS", "SONACOMS.NS", "UNOMINDA.NS", "SCHNEIDER.NS", "HINDCOPPER.NS",
    # Frontline Index Equities
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "ONGC.NS", "KOTAKBANK.NS", "NTPC.NS", "AXISBANK.NS", "TITAN.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJ-AUTO.NS",
    "TECHM.NS", "EICHERMOT.NS", "M&M.NS", "COALINDIA.NS", "HINDALCO.NS",
    "APOLLOHOSP.NS", "WIPRO.NS", "NESTLEIND.NS", "GRASIM.NS", "JSWSTEEL.NS",
    "POWERGRID.NS", "VEDL.NS", "TATASTEEL.NS", "BPCL.NS", "IOC.NS"
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
target_display_count = st.sidebar.slider("Number of Top Stocks to Display", 10, 50, 20, step=5)
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
    all_symbols = list(set(EXPANDED_NSE_UNIVERSE)) + list(SECTOR_MAP.values())
    
    # auto_adjust=False ensures historical dividends/splits don't warp the real-time LTP
    raw = yf.download(
        all_symbols,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    stock_rows = []
    for ticker in EXPANDED_NSE_UNIVERSE:
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
            sma_20 = float(df["Close"].rolling(20).mean().iloc[-1])
            high_20d = float(df["High"].iloc[-21:-1].max()) if len(df) >= 21 else high_p

            open_low_diff_pct = round(((open_p - low_p) / open_p) * 100, 2)
            open_high_diff_pct = round(((high_p - open_p) / open_p) * 100, 2)
            dist_from_50sma_pct = round(((ltp - sma_50) / sma_50) * 100, 2)

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
                "20 SMA": round(sma_20, 2),
                "20D High": round(high_20d, 2),
                "Turnover (Cr)": vol_cr,
                "OpenLow_Diff": open_low_diff_pct,
                "OpenHigh_Diff": open_high_diff_pct,
                "Dist_50SMA": dist_from_50sma_pct
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

st.title("⚡ Dynamic NSE Top 20 Multi-Strategy Screener")
st.caption("Live screening across 150+ equities for Intraday setups, MTF Swings, and Sector Divergence.")

with st.spinner("Downloading unadjusted market quotes..."):
    df_stocks, df_sectors = fetch_complete_snapshot()

if df_stocks.empty:
    st.error("Market data currently unavailable. Please verify internet connectivity and refresh.")
    st.stop()

# Summary KPI Cards
best_sector = df_sectors.sort_values(by="Change (%)", ascending=False).iloc[0] if not df_sectors.empty else None
worst_sector = df_sectors.sort_values(by="Change (%)", ascending=True).iloc[0] if not df_sectors.empty else None

c1, c2, c3 = st.columns(3)
c1.metric("Universe Scanned", f"{len(df_stocks)} Liquid Stocks")
if best_sector is not None:
    c2.metric("Leading Sector", str(best_sector["Sector"]), f"{best_sector['Change (%)']:+.2f}%")
if worst_sector is not None:
    c3.metric("Worst Drag Sector", str(worst_sector["Sector"]), f"{worst_sector['Change (%)']:+.2f}%", delta_color="inverse")

if worst_sector is not None:
    st.error(f"🚨 WEAKEST SECTOR: {worst_sector['Sector']} ({worst_sector['Change (%)']:+.2f}%) — Avoid long setups in this basket.")

# Primary Strategy Tabs
tab_intra, tab_mtf, tab_volatility, tab_overview, tab_sectors = st.tabs([
    f"🔥 Top {target_display_count} Intraday Setups",
    f"🎯 Top {target_display_count} MTF Swings",
    f"⚡ Top {target_display_count} High Volatility Movers",
    "📊 Market Gainers & Losers",
    "🌐 Sector Relative Strength"
])

# ---------------------------------------------------------
# Tab 1: Top Intraday Setups
# ---------------------------------------------------------
with tab_intra:
    st.subheader(f"🔥 Top {target_display_count} Intraday Setups")

    intra_mode = st.radio(
        "Select Intraday Setup Condition:",
        [
            "🟢 Open = Low (Dominant Buyers from Open)",
            "🔴 Open = High (Aggressive Institutional Selling)",
            "🚀 High Relative Volume Breakout (RVOL)",
            "💥 Range Expansion (Day Range % + Holding Pivot)"
        ],
        horizontal=True
    )

    if "Open = Low" in intra_mode:
        sorted_df = df_stocks[(df_stocks["LTP"] >= df_stocks["Open"])].sort_values(
            by=["OpenLow_Diff", "Change (%)"], ascending=[True, False]
        )
        trade_bias = "BUY / LONG"
    elif "Open = High" in intra_mode:
        sorted_df = df_stocks[(df_stocks["LTP"] <= df_stocks["Open"])].sort_values(
            by=["OpenHigh_Diff", "Change (%)"], ascending=[True, True]
        )
        trade_bias = "SELL / SHORT"
    elif "High Relative Volume" in intra_mode:
        sorted_df = df_stocks.sort_values(by=["RVOL", "Turnover (Cr)"], ascending=[False, False])
        trade_bias = "MOMENTUM"
    else:
        sorted_df = df_stocks[(df_stocks["LTP"] > df_stocks["Pivot"])].sort_values(
            by=["Day Range (%)", "Change (%)"], ascending=[False, False]
        )
        trade_bias = "EXPANSION BUY"

    top_intra = sorted_df.head(target_display_count).copy()

    display_cols = [
        "Symbol", "LTP", "Change (%)", "Open", "Day High", "Day Low",
        "RVOL", "Day Range (%)", "Pivot", "Turnover (Cr)"
    ]
    st.dataframe(top_intra[display_cols], use_container_width=True, hide_index=True)

    st.markdown(f"#### 🎯 Top 5 Immediate Action Cards ({trade_bias})")
    for _, row in top_intra.head(5).iterrows():
        ltp = row["LTP"]
        is_long = "BUY" in trade_bias or row["Change (%)"] >= 0

        if is_long:
            sl = min(row["Day Low"], round(ltp * 0.992, 2))
            risk = ltp - sl if (ltp - sl) > 0 else ltp * 0.008
            t1 = round(ltp + (risk * 1.5), 2)
            t2 = round(ltp + (risk * 2.0), 2)
        else:
            sl = max(row["Day High"], round(ltp * 1.008, 2))
            risk = sl - ltp if (sl - ltp) > 0 else ltp * 0.008
            t1 = round(ltp - (risk * 1.5), 2)
            t2 = round(ltp - (risk * 2.0), 2)

        with st.container(border=True):
            head_col, info_col = st.columns([3, 2])
            head_col.markdown(f"### **{row['Symbol']}** — ₹{ltp} ({row['Change (%)']:+.2f}%)")
            info_col.markdown(f"**RVOL:** `{row['RVOL']}x` | **Range:** `{row['Day Range (%)']}%` | **Pivot:** ₹{row['Pivot']}")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Action", "BUY PULLBACK" if is_long else "SELL BOUNCE")
            e2.metric("Target 1 (1:1.5)", f"₹{t1}")
            e3.metric("Target 2 (1:2.0)", f"₹{t2}")
            e4.metric("Strict Stop Loss", f"₹{sl}", delta_color="inverse")

# ---------------------------------------------------------
# Tab 2: Top MTF Swings
# ---------------------------------------------------------
with tab_mtf:
    st.subheader(f"🎯 Top {target_display_count} MTF Positional Swings (4x Leverage)")

    mtf_strat = st.radio(
        "Select MTF Strategy:",
        [
            "1. 50-Day SMA Pullback (Nearest to 50 SMA Support)",
            "2. 20-Day High Breakout + Volume Surge",
            "3. RSI 40-52 Support Reversal (Healthy Pullbacks)"
        ],
        horizontal=True
    )

    if "50-Day SMA Pullback" in mtf_strat:
        df_stocks["SMA_Distance_Abs"] = df_stocks["Dist_50SMA"].abs()
        sorted_mtf = df_stocks[(df_stocks["LTP"] >= df_stocks["50 SMA"] * 0.95)].sort_values(
            by=["SMA_Distance_Abs", "Change (%)"], ascending=[True, False]
        )
    elif "20-Day High Breakout" in mtf_strat:
        df_stocks["Breakout_Proximity"] = round(((df_stocks["Day High"] - df_stocks["20D High"]) / df_stocks["20D High"]) * 100, 2)
        sorted_mtf = df_stocks.sort_values(by=["Breakout_Proximity", "RVOL"], ascending=[False, False])
    else:
        sorted_mtf = df_stocks[(df_stocks["RSI (14)"] >= 35) & (df_stocks["RSI (14)"] <= 55)].sort_values(
            by=["RSI (14)", "Change (%)"], ascending=[True, False]
        )

    top_mtf = sorted_mtf.head(target_display_count).copy()

    st.dataframe(
        top_mtf[["Symbol", "LTP", "Change (%)", "50 SMA", "Dist_50SMA", "RSI (14)", "RVOL", "Turnover (Cr)"]],
        use_container_width=True, hide_index=True
    )

    st.markdown("#### 🛒 Recommended MTF Execution Orders")
    for _, row in top_mtf.head(5).iterrows():
        ltp = row["LTP"]
        margin = round(ltp * 0.22, 2)
        t1 = round(ltp * 1.055, 2)
        t2 = round(ltp * 1.10, 2)
        sl = round(ltp * 0.965, 2)
        with st.container(border=True):
            st.markdown(f"**{row['Symbol']}** — LTP: ₹{ltp} ({row['Change (%)']:+.2f}%) | Est. Margin: ~₹{margin}/sh")
            st.write(f"Target 1 (+5.5%): ₹{t1} | Target 2 (+10%): ₹{t2} | Stop Loss (-3.5%): ₹{sl} | 50 SMA: ₹{row['50 SMA']}")

# ---------------------------------------------------------
# Tab 3: Top High Volatility
# ---------------------------------------------------------
with tab_volatility:
    st.subheader(f"⚡ Top {target_display_count} High Volatility & Momentum Equities")
    st.caption("Ranked by Day Range Expansion & Turnover. Best for active intraday range riders.")
    top_vol = df_stocks.sort_values(by=["Day Range (%)", "Turnover (Cr)"], ascending=[False, False]).head(target_display_count)
    st.dataframe(
        top_vol[["Symbol", "LTP", "Change (%)", "Day Range (%)", "Day High", "Day Low", "RVOL", "Turnover (Cr)"]],
        use_container_width=True, hide_index=True
    )

# ---------------------------------------------------------
# Tab 4: Overview
# ---------------------------------------------------------
with tab_overview:
    cg, cl = st.columns(2)
    with cg:
        st.subheader(f"🟢 Top {target_display_count} Gainers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=False).head(target_display_count)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)
    with cl:
        st.subheader(f"🔴 Top {target_display_count} Losers")
        st.dataframe(df_stocks.sort_values(by="Change (%)", ascending=True).head(target_display_count)[["Symbol", "LTP", "Change (%)", "Pivot", "Turnover (Cr)"]], use_container_width=True, hide_index=True)

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