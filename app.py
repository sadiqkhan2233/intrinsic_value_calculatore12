"""
Intrinsic Value Calculator (DCF) - Streamlit app
Data: yfinance | Charts: Plotly | Optional AI summary: Groq API
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DCF Intrinsic Value Calculator",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.25);
            padding: 16px 20px;
            border-radius: 12px;
        }
        .verdict {
            padding: 14px 18px; border-radius: 12px; font-size: 1.05rem;
            font-weight: 600; margin: 8px 0 18px 0;
        }
        .under {background: rgba(0,170,90,0.15); border: 1px solid rgba(0,170,90,0.5);}
        .over  {background: rgba(220,50,50,0.15); border: 1px solid rgba(220,50,50,0.5);}
        .fair  {background: rgba(240,170,0,0.15); border: 1px solid rgba(240,170,0,0.5);}
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECTION_YEARS = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


class DataError(Exception):
    """Raised for problems with ticker data that should be shown to the user."""


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
def _row(df: pd.DataFrame, names: list):
    """Return the first matching row (as numeric Series) from a yfinance statement."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce")
    return None


def _latest(series):
    """Latest non-null value of a statement row (columns are newest-first)."""
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float(s.sort_index().iloc[-1])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_data(ticker: str) -> dict:
    """Pull everything we need from yfinance and return plain Python/pandas objects."""
    tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    cf = tk.cashflow
    bs = tk.balance_sheet
    inc = tk.income_stmt

    if cf is None or cf.empty:
        raise DataError(
            f"No cash flow statement found for '{ticker}'. "
            "Check the ticker symbol (funds, ETFs and some foreign listings have no statements)."
        )

    # --- Free Cash Flow = Operating Cash Flow - CapEx -------------------------
    ocf = _row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities",
                    "Cash Flow From Continuing Operating Activities"])
    capex = _row(cf, ["Capital Expenditure", "Capital Expenditures",
                      "Purchase Of PPE", "Net PPE Purchase And Sale"])

    if ocf is not None and capex is not None:
        fcf = (ocf - capex.abs()).dropna()
    else:
        fcf = _row(cf, ["Free Cash Flow"])
        fcf = fcf.dropna() if fcf is not None else None

    if fcf is None or fcf.empty:
        raise DataError("Could not find Operating Cash Flow / CapEx data to compute Free Cash Flow.")
    fcf = fcf.sort_index()  # oldest -> newest

    # --- Debt, Cash, Shares ----------------------------------------------------
    debt = _latest(_row(bs, ["Total Debt"]))
    if debt is None:
        lt = _latest(_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])) or 0.0
        st_debt = _latest(_row(bs, ["Current Debt", "Current Debt And Capital Lease Obligation"])) or 0.0
        debt = lt + st_debt
    if debt is None:
        debt = float(info.get("totalDebt") or 0.0)

    cash = _latest(_row(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                             "Cash And Short Term Investments"]))
    if cash is None:
        cash = float(info.get("totalCash") or 0.0)

    shares = _latest(_row(inc, ["Diluted Average Shares", "Diluted Shares Outstanding"]))
    if not shares:
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if not shares:
        raise DataError("Could not determine diluted shares outstanding.")

    # --- Current price ---------------------------------------------------------
    price = None
    try:
        price = tk.fast_info.get("last_price")
    except Exception:
        pass
    if not price:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        try:
            hist = tk.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass
    if not price:
        raise DataError("Could not fetch the current market price.")

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "USD"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "price": float(price),
        "fcf": fcf,
        "debt": float(debt),
        "cash": float(cash),
        "shares": float(shares),
    }


# ----------------------------------------------------------------------------
# DCF engine
# ----------------------------------------------------------------------------
def historical_fcf_cagr(fcf: pd.Series):
    """Return (bounded_cagr, raw_cagr_or_None). Bounded to [2%, 15%]."""
    values = fcf.values
    n = len(values) - 1
    if n >= 1 and values[0] > 0 and values[-1] > 0:
        raw = (values[-1] / values[0]) ** (1 / n) - 1
        return float(np.clip(raw, 0.02, 0.15)), float(raw)
    return 0.05, None  # fallback when CAGR is undefined (negative/zero FCF endpoints)


def run_dcf(fcf0, growth, wacc, tg, debt, cash, shares, years=PROJECTION_YEARS):
    """Run a single-scenario DCF. Returns dict of results."""
    if wacc <= tg:
        raise DataError("Discount rate (WACC) must be greater than the terminal growth rate.")

    t = np.arange(1, years + 1)
    projected = fcf0 * (1 + growth) ** t
    discount = (1 + wacc) ** t
    pv_fcf = projected / discount

    terminal_value = projected[-1] * (1 + tg) / (wacc - tg)
    pv_terminal = terminal_value / (1 + wacc) ** years

    enterprise_value = pv_fcf.sum() + pv_terminal
    equity_value = enterprise_value - debt + cash
    per_share = equity_value / shares

    return {
        "growth": growth,
        "projected": projected,
        "pv_fcf": pv_fcf,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "per_share": per_share,
    }


# ----------------------------------------------------------------------------
# Groq AI summary
# ----------------------------------------------------------------------------
def groq_summary(api_key: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_tokens": 400,
        "messages": [
            {"role": "system",
             "content": "You are a concise, careful equity analyst. Respond with exactly 3 bullet points. "
                        "No preamble, no investment advice, no made-up facts."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code == 401:
        raise DataError("Groq rejected the API key (401 Unauthorized).")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def fmt_big(x, cur="$"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if x >= div:
            return f"{sign}{cur}{x / div:,.2f}{unit}"
    return f"{sign}{cur}{x:,.0f}"


def fmt_price(x, cur="$"):
    return f"{cur}{x:,.2f}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Inputs")
ticker_input = st.sidebar.text_input("Stock ticker", value="AAPL", help="e.g. AAPL, MSFT, TSLA").strip().upper()
wacc_pct = st.sidebar.slider("Discount rate (WACC) %", 5.0, 15.0, 9.0, 0.1)
tg_pct = st.sidebar.slider("Terminal growth rate %", 0.0, 5.0, 2.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Groq AI (optional)")

groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")  # Streamlit Cloud secrets
    except Exception:
        groq_key = ""
if groq_key:
    st.sidebar.success("GROQ_API_KEY detected.")
else:
    groq_key = st.sidebar.text_input("Groq API key", type="password",
                                     help="Not stored. Or set the GROQ_API_KEY environment variable / secret.")

st.sidebar.caption("Educational tool only. Not investment advice.")

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("📈 Intrinsic Value Calculator")
st.caption("Discounted Cash Flow (DCF) valuation with Bear / Base / Bull scenarios")

if not ticker_input:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        data = fetch_company_data(ticker_input)
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load data for '{ticker_input}'. It may be an invalid ticker or a connection "
             f"problem with Yahoo Finance. Details: {e}")
    st.stop()

cur_code = data["currency"]
cur = "$" if cur_code == "USD" else f"{cur_code} "
fcf_hist: pd.Series = data["fcf"]
fcf0 = float(fcf_hist.iloc[-1])

if fcf0 <= 0:
    st.error(
        f"{data['name']} has negative or zero latest Free Cash Flow ({fmt_big(fcf0, cur)}). "
        "A standard FCF-based DCF is not meaningful for this company."
    )
    st.stop()

base_growth, raw_cagr = historical_fcf_cagr(fcf_hist)
wacc, tg = wacc_pct / 100, tg_pct / 100

scenarios = {
    "Bear": base_growth * 0.7,
    "Base": base_growth,
    "Bull": base_growth * 1.3,
}

try:
    results = {
        name: run_dcf(fcf0, g, wacc, tg, data["debt"], data["cash"], data["shares"])
        for name, g in scenarios.items()
    }
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"DCF calculation failed: {e}")
    st.stop()

price = data["price"]
base = results["Base"]
upside = base["per_share"] / price - 1

# --- Header ------------------------------------------------------------------
st.subheader(f"{data['name']} ({data['ticker']})")
st.caption(f"{data['sector']} · {data['industry']} · Market cap: {fmt_big(data['market_cap'], cur)}")

# --- Hero metrics ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Current Market Price", fmt_price(price, cur))
c2.metric("Base Intrinsic Value", fmt_price(base["per_share"], cur), delta=f"{upside:+.1%} vs. market")
c3.metric("Base Growth Rate (FCF CAGR)", f"{base_growth:.2%}",
          help="Historical FCF CAGR, bounded between 2% and 15%.")

if upside > 0.10:
    st.markdown(f'<div class="verdict under">🟢 Potentially UNDERVALUED: base-case value is '
                f'{upside:.1%} above the market price.</div>', unsafe_allow_html=True)
elif upside < -0.10:
    st.markdown(f'<div class="verdict over">🔴 Potentially OVERVALUED: market price is '
                f'{abs(upside):.1%} above base-case value.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="verdict fair">🟡 Roughly FAIRLY VALUED (within ±10% of base-case value).</div>',
                unsafe_allow_html=True)

if raw_cagr is None:
    st.warning("Historical FCF CAGR could not be computed (non-positive endpoints). Using a 5% default growth rate.")
elif raw_cagr != base_growth:
    st.info(f"Raw historical FCF CAGR was {raw_cagr:.2%}; bounded to {base_growth:.2%} (allowed range 2%–15%).")

# --- Scenario table ----------------------------------------------------------
st.markdown("### Scenario Comparison")
table = pd.DataFrame({
    "Scenario": list(results.keys()),
    "FCF Growth": [f"{r['growth']:.2%}" for r in results.values()],
    "Enterprise Value": [fmt_big(r["enterprise_value"], cur) for r in results.values()],
    "Equity Value": [fmt_big(r["equity_value"], cur) for r in results.values()],
    "Intrinsic Value / Share": [fmt_price(r["per_share"], cur) for r in results.values()],
    "Upside / Downside": [f"{r['per_share'] / price - 1:+.1%}" for r in results.values()],
})
st.dataframe(table, hide_index=True, use_container_width=True)

# --- Charts ------------------------------------------------------------------
st.markdown("### Free Cash Flow")
col_a, col_b = st.columns([3, 2])

years_labels = [f"Year {i}" for i in range(1, PROJECTION_YEARS + 1)]
fig = go.Figure()
fig.add_bar(x=years_labels, y=base["projected"], name="Base (projected FCF)", marker_color="#2E86DE")
fig.add_bar(x=years_labels, y=base["pv_fcf"], name="Base (present value)", marker_color="#A5C8F0")
fig.add_scatter(x=years_labels, y=results["Bear"]["projected"], mode="lines+markers",
                name="Bear", line=dict(color="#E74C3C", dash="dash"))
fig.add_scatter(x=years_labels, y=results["Bull"]["projected"], mode="lines+markers",
                name="Bull", line=dict(color="#27AE60", dash="dash"))
fig.update_layout(
    title=f"{PROJECTION_YEARS}-Year Projected FCF",
    barmode="group", height=420, yaxis_title=f"FCF ({cur_code})",
    legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10),
)
col_a.plotly_chart(fig, use_container_width=True)

fig_h = go.Figure(go.Bar(
    x=[d.strftime("%Y") if hasattr(d, "strftime") else str(d) for d in fcf_hist.index],
    y=fcf_hist.values,
    marker_color=["#27AE60" if v >= 0 else "#E74C3C" for v in fcf_hist.values],
))
fig_h.update_layout(title="Historical FCF (OCF − CapEx)", height=420, yaxis_title=f"FCF ({cur_code})",
                    margin=dict(l=10, r=10, t=50, b=10))
col_b.plotly_chart(fig_h, use_container_width=True)

# --- Details -----------------------------------------------------------------
with st.expander("🔍 Valuation inputs & Base Case breakdown"):
    d1, d2 = st.columns(2)
    d1.markdown(
        f"""
        **Inputs**
        - Latest FCF: {fmt_big(fcf0, cur)}
        - Total debt: {fmt_big(data['debt'], cur)}
        - Cash & equivalents: {fmt_big(data['cash'], cur)}
        - Diluted shares: {data['shares'] / 1e6:,.2f}M
        - WACC: {wacc:.2%} · Terminal growth: {tg:.2%}
        """
    )
    d2.markdown(
        f"""
        **Base Case**
        - PV of 5-yr FCF: {fmt_big(base['pv_fcf'].sum(), cur)}
        - Terminal value (undiscounted): {fmt_big(base['terminal_value'], cur)}
        - PV of terminal value: {fmt_big(base['pv_terminal'], cur)}
        - Enterprise value: {fmt_big(base['enterprise_value'], cur)}
        - Equity value: {fmt_big(base['equity_value'], cur)}
        - Terminal value share of EV: {base['pv_terminal'] / base['enterprise_value']:.0%}
        """
    )

# --- Groq AI section ---------------------------------------------------------
st.markdown("### 🤖 AI Financial Health Summary")
if not groq_key:
    st.info("Add a Groq API key in the sidebar (or set the `GROQ_API_KEY` environment variable) "
            "to enable the AI summary. The DCF above works without it.")
else:
    if st.button("Generate AI summary"):
        prompt = (
            f"Company: {data['name']} ({data['ticker']}), sector: {data['sector']}, industry: {data['industry']}.\n"
            f"Market cap: {fmt_big(data['market_cap'], cur)}; trailing P/E: {data['trailing_pe']}.\n"
            f"Latest FCF: {fmt_big(fcf0, cur)}; FCF CAGR (bounded): {base_growth:.2%}.\n"
            f"Total debt: {fmt_big(data['debt'], cur)}; cash: {fmt_big(data['cash'], cur)}.\n"
            f"Price: {fmt_price(price, cur)}; DCF base intrinsic value: {fmt_price(base['per_share'], cur)} "
            f"({upside:+.1%}); bear {fmt_price(results['Bear']['per_share'], cur)}, "
            f"bull {fmt_price(results['Bull']['per_share'], cur)}.\n"
            "Give a brief 3-bullet qualitative financial health summary (cash generation, balance sheet, "
            "valuation vs. DCF)."
        )
        try:
            with st.spinner("Asking Groq..."):
                st.markdown(groq_summary(groq_key, prompt))
        except DataError as e:
            st.error(str(e))
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the Groq API: {e}")
        except Exception as e:
            st.error(f"Unexpected error generating AI summary: {e}")

st.markdown("---")
st.caption("Data from Yahoo Finance via yfinance (may be delayed or incomplete). DCF output is highly sensitive "
           "to growth and discount-rate assumptions. For educational purposes only, not financial advice.")
