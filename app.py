"""
PSX Intrinsic Value Calculator (DCF) - Streamlit app
Pakistan Stock Exchange only | Values in PKR
Data: yfinance (.KA tickers) | Charts: Plotly | Optional AI summary: Groq API
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PSX DCF Calculator", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.25);
            padding: 16px 20px; border-radius: 12px;
        }
        .verdict {padding: 14px 18px; border-radius: 12px; font-size: 1.05rem;
                  font-weight: 600; margin: 8px 0 18px 0;}
        .under {background: rgba(0,170,90,0.15); border: 1px solid rgba(0,170,90,0.5);}
        .over  {background: rgba(220,50,50,0.15); border: 1px solid rgba(220,50,50,0.5);}
        .fair  {background: rgba(240,170,0,0.15); border: 1px solid rgba(240,170,0,0.5);}
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECTION_YEARS = 5
CUR = "Rs "  # all values shown in Pakistani Rupees
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Starter list of major PSX symbols (Yahoo Finance format = SYMBOL.KA).
# Some may be renamed/delisted over time; the app handles missing data gracefully.
DEFAULT_PSX_SYMBOLS = """
OGDC PPL POL MARI PSO SNGP SSGC APL ATRL NRL PRL CNERGY SHEL HTL
HBL UBL MCB BAFL BAHL MEBL NBP ABL AKBL FABL BOP BOK SNBL JSBL HMB SMBL
LUCK DGKC MLCF FCCL CHCC KOHC PIOC ACPL POWER GWLC BWCL
ENGRO EFERT FFC FFBL FATIMA AGL
HUBC KAPCO NPL NCPL KEL PKGP LPL
SYS NETSOL TRG AVN
SEARL AGP GLAXO HINOON ABOT FEROZ
NML NCL GATM ILP KTML
INDU HCAR PSMC MTL AGTL ATLH GHNI
ICI LOTCHEM EPCL COLG
NESTLE UNITY FFL TREET MUREB
PAEL PIBTL PTC TELE WTL NATF PKGS CEPB THALL DAWH SPWL AICL JGICL EFUG
ISL ASTL ASL MUGHAL ITTEFAQ SAZEW GHGL GADT TGL EPQL
""".split()


class DataError(Exception):
    """Problems with ticker data that should be shown to the user."""


@st.cache_data(show_spinner=False)
def load_symbols() -> list:
    symbols = set(DEFAULT_PSX_SYMBOLS)
    if os.path.exists("psx_symbols.txt"):
        try:
            with open("psx_symbols.txt", "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip().upper().replace(".KA", "")
                    if s and not s.startswith("#"):
                        symbols.add(s)
        except Exception:
            pass
    return sorted(symbols)


def normalize_symbol(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "")
    if s.endswith(".KA"):
        s = s[:-3]
    return s


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
def _row(df, names):
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce")
    return None


def _latest(series):
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float(s.sort_index().iloc[-1])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_data(symbol: str) -> dict:
    yf_ticker = f"{symbol}.KA"
    tk = yf.Ticker(yf_ticker)

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    cf = tk.cashflow
    bs = tk.balance_sheet
    inc = tk.income_stmt

    if cf is None or cf.empty:
        raise DataError(
            f"No cash flow statement found on Yahoo Finance for '{yf_ticker}'. "
            "Check the PSX symbol, or the company may not have statement coverage on Yahoo."
        )

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
    fcf = fcf.sort_index()

    debt = _latest(_row(bs, ["Total Debt"]))
    if debt is None:
        lt = _latest(_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]))
        sd = _latest(_row(bs, ["Current Debt", "Current Debt And Capital Lease Obligation"]))
        if lt is not None or sd is not None:
            debt = (lt or 0.0) + (sd or 0.0)
        else:
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
        "symbol": symbol,
        "yf_ticker": yf_ticker,
        "name": info.get("longName") or info.get("shortName") or symbol,
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
    values = fcf.values
    n = len(values) - 1
    if n >= 1 and values[0] > 0 and values[-1] > 0:
        raw = (values[-1] / values[0]) ** (1 / n) - 1
        return float(np.clip(raw, 0.02, 0.15)), float(raw)
    return 0.05, None


def run_dcf(fcf0, growth, wacc, tg, debt, cash, shares, years=PROJECTION_YEARS):
    if wacc <= tg:
        raise DataError("Discount rate (WACC) must be greater than the terminal growth rate.")

    t = np.arange(1, years + 1)
    projected = fcf0 * (1 + growth) ** t
    pv_fcf = projected / (1 + wacc) ** t

    terminal_value = projected[-1] * (1 + tg) / (wacc - tg)
    pv_terminal = terminal_value / (1 + wacc) ** years

    enterprise_value = pv_fcf.sum() + pv_terminal
    equity_value = enterprise_value - debt + cash
    per_share = equity_value / shares

    return {
        "growth": growth, "projected": projected, "pv_fcf": pv_fcf,
        "terminal_value": terminal_value, "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value, "equity_value": equity_value,
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
             "content": "You are a concise, careful equity analyst covering the Pakistan Stock Exchange. "
                        "Respond with exactly 3 bullet points. No preamble, no investment advice, "
                        "no made-up facts."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code == 401:
        raise DataError("Groq rejected the API key (401 Unauthorized).")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
# Formatting helpers (PKR)
# ----------------------------------------------------------------------------
def fmt_big(x, cur=CUR):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if x >= div:
            return f"{sign}{cur}{x / div:,.2f}{unit}"
    return f"{sign}{cur}{x:,.0f}"


def fmt_price(x, cur=CUR):
    return f"{cur}{x:,.2f}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Inputs")
symbols = load_symbols()

default_idx = symbols.index("OGDC") if "OGDC" in symbols else 0
picked = st.sidebar.selectbox(f"PSX stock ({len(symbols)} listed - type to search)", symbols, index=default_idx)
custom = st.sidebar.text_input("Or type any PSX symbol", value="",
                               help="e.g. HUBC, LUCK, ENGRO. '.KA' is added automatically.")
symbol = normalize_symbol(custom) if custom.strip() else picked

wacc_pct = st.sidebar.slider("Discount rate (WACC) %", 8.0, 30.0, 16.0, 0.5,
                             help="Pakistan rates are much higher than US rates. Check current T-bill / PIB yields.")
tg_pct = st.sidebar.slider("Terminal growth rate %", 0.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Groq AI (optional)")

groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
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
st.title("📈 PSX Intrinsic Value Calculator")
st.caption("Discounted Cash Flow (DCF) valuation for Pakistan Stock Exchange stocks · All values in PKR")

if not symbol:
    st.info("Select or type a PSX symbol in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Fetching data for {symbol}.KA ..."):
        data = fetch_company_data(symbol)
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load data for '{symbol}.KA'. It may be an invalid symbol or a connection "
             f"problem with Yahoo Finance. Details: {e}")
    st.stop()

fcf_hist: pd.Series = data["fcf"]
fcf0 = float(fcf_hist.iloc[-1])

if "financial" in str(data["sector"]).lower():
    st.warning("This looks like a bank / financial company. FCF-based DCF is generally not meaningful for "
               "banks (operating cash flow includes deposit and loan flows). Treat the result with caution; "
               "dividend-discount or P/B methods suit them better.")

if fcf0 <= 0:
    st.error(
        f"{data['name']} has negative or zero latest Free Cash Flow ({fmt_big(fcf0)}). "
        "A standard FCF-based DCF is not meaningful for this company."
    )
    st.stop()

base_growth, raw_cagr = historical_fcf_cagr(fcf_hist)
wacc, tg = wacc_pct / 100, tg_pct / 100

scenarios = {"Bear": base_growth * 0.7, "Base": base_growth, "Bull": base_growth * 1.3}

try:
    results = {n: run_dcf(fcf0, g, wacc, tg, data["debt"], data["cash"], data["shares"])
               for n, g in scenarios.items()}
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"DCF calculation failed: {e}")
    st.stop()

price = data["price"]
base = results["Base"]
upside = base["per_share"] / price - 1

st.subheader(f"{data['name']} ({data['symbol']})")
st.caption(f"{data['sector']} · {data['industry']} · Market cap: {fmt_big(data['market_cap'])}")

c1, c2, c3 = st.columns(3)
c1.metric("Current Market Price", fmt_price(price))
c2.metric("Base Intrinsic Value", fmt_price(base["per_share"]), delta=f"{upside:+.1%} vs. market")
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

if base["per_share"] > price * 5 or base["per_share"] < price * 0.2:
    st.warning("The intrinsic value is extremely far from the market price. This usually means a data issue "
               "(one-off cash flow year, unit mismatch, missing debt) or unrealistic inputs. Double-check "
               "the numbers in the breakdown below.")

if raw_cagr is None:
    st.warning("Historical FCF CAGR could not be computed (non-positive endpoints). Using a 5% default growth rate.")
elif raw_cagr != base_growth:
    st.info(f"Raw historical FCF CAGR was {raw_cagr:.2%}; bounded to {base_growth:.2%} (allowed range 2%–15%).")

st.markdown("### Scenario Comparison")
table = pd.DataFrame({
    "Scenario": list(results.keys()),
    "FCF Growth": [f"{r['growth']:.2%}" for r in results.values()],
    "Enterprise Value": [fmt_big(r["enterprise_value"]) for r in results.values()],
    "Equity Value": [fmt_big(r["equity_value"]) for r in results.values()],
    "Intrinsic Value / Share": [fmt_price(r["per_share"]) for r in results.values()],
    "Upside / Downside": [f"{r['per_share'] / price - 1:+.1%}" for r in results.values()],
})
st.dataframe(table, hide_index=True, use_container_width=True)

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
fig.update_layout(title=f"{PROJECTION_YEARS}-Year Projected FCF", barmode="group", height=420,
                  yaxis_title="FCF (PKR)", legend=dict(orientation="h", y=-0.2),
                  margin=dict(l=10, r=10, t=50, b=10))
col_a.plotly_chart(fig, use_container_width=True)

fig_h = go.Figure(go.Bar(
    x=[d.strftime("%Y") if hasattr(d, "strftime") else str(d) for d in fcf_hist.index],
    y=fcf_hist.values,
    marker_color=["#27AE60" if v >= 0 else "#E74C3C" for v in fcf_hist.values],
))
fig_h.update_layout(title="Historical FCF (OCF − CapEx)", height=420, yaxis_title="FCF (PKR)",
                    margin=dict(l=10, r=10, t=50, b=10))
col_b.plotly_chart(fig_h, use_container_width=True)

with st.expander("🔍 Valuation inputs & Base Case breakdown"):
    d1, d2 = st.columns(2)
    d1.markdown(
        f"""
        **Inputs**
        - Latest FCF: {fmt_big(fcf0)}
        - Total debt: {fmt_big(data['debt'])}
        - Cash & equivalents: {fmt_big(data['cash'])}
        - Diluted shares: {data['shares'] / 1e6:,.2f}M
        - WACC: {wacc:.2%} · Terminal growth: {tg:.2%}
        """
    )
    d2.markdown(
        f"""
        **Base Case**
        - PV of 5-yr FCF: {fmt_big(base['pv_fcf'].sum())}
        - Terminal value (undiscounted): {fmt_big(base['terminal_value'])}
        - PV of terminal value: {fmt_big(base['pv_terminal'])}
        - Enterprise value: {fmt_big(base['enterprise_value'])}
        - Equity value: {fmt_big(base['equity_value'])}
        - Terminal value share of EV: {base['pv_terminal'] / base['enterprise_value']:.0%}
        """
    )

st.markdown("### 🤖 AI Financial Health Summary")
if not groq_key:
    st.info("Add a Groq API key in the sidebar (or set the `GROQ_API_KEY` environment variable) "
            "to enable the AI summary. The DCF above works without it.")
else:
    if st.button("Generate AI summary"):
        prompt = (
            f"Company: {data['name']} (PSX: {data['symbol']}), sector: {data['sector']}, "
            f"industry: {data['industry']}. All amounts are in Pakistani Rupees (PKR).\n"
            f"Market cap: {fmt_big(data['market_cap'])}; trailing P/E: {data['trailing_pe']}.\n"
            f"Latest FCF: {fmt_big(fcf0)}; FCF CAGR (bounded): {base_growth:.2%}.\n"
            f"Total debt: {fmt_big(data['debt'])}; cash: {fmt_big(data['cash'])}.\n"
            f"Price: {fmt_price(price)}; DCF base intrinsic value: {fmt_price(base['per_share'])} "
            f"({upside:+.1%}); bear {fmt_price(results['Bear']['per_share'])}, "
            f"bull {fmt_price(results['Bull']['per_share'])}.\n"
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
st.caption("Data from Yahoo Finance via yfinance (PSX coverage can be delayed or incomplete). DCF output is "
           "highly sensitive to growth and discount-rate assumptions. Educational use only, not financial advice.")
