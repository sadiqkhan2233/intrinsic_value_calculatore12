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

    shares = _latest(_row(inc, ["Diluted Average Shares", "Diluted
