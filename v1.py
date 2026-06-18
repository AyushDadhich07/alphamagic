"""
AlphaMagic — AI-Assisted Indian Stock Research MVP
"Find useful signals faster — verify before investing"

MVP POSITIONING:
  This is a demo/research-assistant dashboard for Indian equities.
  It uses yfinance for now. Data can be incomplete or stale, especially for
  Indian fundamentals and governance fields. Treat every output as a triage
  signal, not an investment recommendation.

MARKET GAP THIS SOLVES:
  Screener.in = loved by experts, too complex for many retail users
  Broker apps = easy but limited research depth
  Moneycontrol = news-heavy, signal-light
  Trendlyne = data-rich, can feel overwhelming
  ──────────────────────────────────────────────
  AlphaMagic MVP = AI-assisted · Nifty 500 max scan universe · Beginner-readable research depth

CORE MVP DIFFERENTIATORS:
  ✦ Nifty 500 scan universe — controlled MVP scope for speed and reliability
  ✦ Any-stock search — find NSE/BSE tickers through yfinance search
  ✦ Sector Alpha Score™ — sector-aware stock score; uses different metrics for financials, IT, pharma, industrials, consumer and cyclicals
  ✦ CG-Proxy Score™     — preliminary governance-risk proxy, not a definitive governance rating
  ✦ EQ-Score™           — earnings-quality proxy based on cash conversion/accruals where data exists
  ✦ Data Availability — shows how much of the required data was actually available
  ✦ Technical Engine  — RSI, MACD, SMA50/200, Bollinger Band signals
  ✦ AI Screener       — natural-language stock discovery
  ✦ Risk Radar        — preliminary risk-signal scan; not a fraud detector

PAGES:
  🏠 Market Pulse   — NSE index snapshot, actual universe movers, 5-day sector proxy returns
  🤖 AI Screener    — "Find quality IT stocks with low debt" → results
  🔬 Deep Dive      — Sector Alpha Score, fundamentals, CG-Proxy, EQ-Score, technicals, cautious AI thesis
  💼 Portfolio Lab  — Health score, historical optimizer, SIP planner
  🚨 Risk Radar     — Preliminary governance + earnings-quality risk signals
  📘 Metrics Guide  — Formulas, interpretation, and limitations

Run: streamlit run alphamagic.py
Deps: pip install streamlit yfinance pandas numpy plotly scipy
AI features: Groq API key (free at groq.com) OR Anthropic API key
"""

import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, re, math

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaMagic · Indian Stock Research",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: #050710; }
section[data-testid="stSidebar"] { display: none !important; }

div[data-testid="metric-container"] {
    background: #0c0e1c; border: 1px solid #1a1e35; border-radius: 12px; padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #3d4466 !important; font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e6ff !important; font-size: 22px !important; font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important; font-family: 'JetBrains Mono', monospace !important;
}

h1, h2, h3 { color: #e2e6ff !important; }
h1 { font-size: 26px !important; font-weight: 900 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 20px !important; font-weight: 700 !important; }
h3 { font-size: 16px !important; font-weight: 600 !important; }
hr { border-color: #1a1e35 !important; margin: 20px 0 !important; }

.stDataFrame { border: 1px solid #1a1e35 !important; border-radius: 10px !important; }
.stButton > button {
    font-size: 13px !important; font-weight: 600 !important; border-radius: 8px !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    border: none !important; box-shadow: 0 0 16px rgba(37,99,235,0.25) !important;
}
.stButton > button:hover { opacity: 0.92 !important; transform: translateY(-1px) !important; }

/* Flags */
.flag-r { background:#160508; border-left:3px solid #ef4444; border-radius:0 8px 8px 0; padding:9px 14px; margin:3px 0; font-size:13px; color:#fca5a5; line-height:1.5; }
.flag-g { background:#04120b; border-left:3px solid #22c55e; border-radius:0 8px 8px 0; padding:9px 14px; margin:3px 0; font-size:13px; color:#86efac; line-height:1.5; }
.flag-y { background:#141008; border-left:3px solid #eab308; border-radius:0 8px 8px 0; padding:9px 14px; margin:3px 0; font-size:13px; color:#fde047; line-height:1.5; }

/* AI box */
.ai-box { background:#060c20; border:1px solid #1e3a8a; border-radius:14px; padding:20px 24px; color:#c7d2fe; font-size:14px; line-height:1.9; }

/* Score gauge */
.gauge-wrap { background:#0c0e1c; border:1px solid #1a1e35; border-radius:14px; padding:20px; text-align:center; }
.gauge-score { font-size:52px; font-weight:900; font-family:'JetBrains Mono',monospace; line-height:1; }
.gauge-grade { font-size:14px; font-weight:700; letter-spacing:0.15em; margin-top:4px; }
.gauge-label { font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#3d4466; margin-top:8px; }

/* Sector card */
.sector-card { background:#0c0e1c; border:1px solid #1a1e35; border-radius:10px; padding:12px 14px; margin:4px 0; }

/* Pills */
.pill { display:inline-block; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; margin:2px; }
.pill-b { background:#0a1530; border:1px solid #2563eb; color:#60a5fa; }
.pill-g { background:#041510; border:1px solid #16a34a; color:#4ade80; }
.pill-r { background:#160505; border:1px solid #dc2626; color:#f87171; }
.pill-y { background:#140f05; border:1px solid #ca8a04; color:#facc15; }
.pill-p { background:#0d0a1e; border:1px solid #7c3aed; color:#a78bfa; }

.lbl { font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#2e3350; display:block; margin-bottom:8px; }
.mono { font-family:'JetBrains Mono',monospace; }

/* Inputs */
.stSelectbox > div > div { background:#0c0e1c !important; border-color:#1a1e35 !important; }
.stTextInput input, .stTextArea textarea { background:#0c0e1c !important; border-color:#1a1e35 !important; color:#e2e6ff !important; border-radius:8px !important; }
.stNumberInput input { background:#0c0e1c !important; border-color:#1a1e35 !important; color:#e2e6ff !important; }
.stTabs [data-baseweb="tab"] { color:#3d4466 !important; font-size:13px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { color:#e2e6ff !important; }
.stRadio > div { gap: 4px !important; }

/* Nav */
.nav-bar { background:#0c0e1c; border-bottom:1px solid #1a1e35; padding:12px 0; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ── NSE Universe ─────────────────────────────────────────────────────────────
NIFTY_200 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BHARTIARTL.NS","BPCL.NS",
    "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS",
    "SBIN.NS","SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","ZOMATO.NS",
    "PERSISTENT.NS","LTIM.NS","COFORGE.NS","MPHASIS.NS","OFSS.NS","TATAELXSI.NS","KPIT.NS",
    "CHOLAFIN.NS","MUTHOOTFIN.NS","BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS",
    "RECLTD.NS","PFC.NS","CAMS.NS","CDSL.NS","MCX.NS",
    "COLPAL.NS","MARICO.NS","DABUR.NS","GODREJCP.NS","TRENT.NS","PAGEIND.NS",
    "TORNTPHARM.NS","LUPIN.NS","ALKEM.NS","ZYDUSLIFE.NS","ABBOTINDIA.NS",
    "SIEMENS.NS","ABB.NS","HAL.NS","BEL.NS","HAVELLS.NS","POLYCAB.NS",
    "COCHINSHIP.NS","MAZDOCK.NS","CGPOWER.NS","BHEL.NS",
    "PIDILITIND.NS","ASTRAL.NS","BERGEPAINT.NS","SUPREMEIND.NS",
    "IOC.NS","GAIL.NS","HINDPETRO.NS","PETRONET.NS","IRCTC.NS","RVNL.NS",
    "DMART.NS","NAUKRI.NS","BATAINDIA.NS","RELAXO.NS",
    "APLAPOLLO.NS","JSPL.NS","HINDZINC.NS",
]

# ── Nifty 500 Extension (Midcap + Smallcap quality names not in Nifty 200) ──
NIFTY_500_EXT = [
    # New-age Tech / IT Midcap
    "ANGELONE.NS","NAZARA.NS","MAPMYINDIA.NS","TANLA.NS","HAPPSTMNDS.NS",
    "MASTEK.NS","SONATSOFTW.NS","BSOFT.NS","ROUTE.NS","NEWGEN.NS","ZENSAR.NS",
    # Midcap Finance / Wealth
    "IIFL.NS","MFSL.NS","MOTILALOFS.NS","KFINTECH.NS","NUVAMA.NS",
    "360ONE.NS","SPANDANA.NS","CREDITACC.NS","UJJIVAN.NS","AUBANK.NS",
    "EQUITASBNK.NS","CANFINHOME.NS","LICHSGFIN.NS","FUSION.NS",
    # Consumer / New-age
    "NYKAA.NS","DELHIVERY.NS","CARTRADE.NS","EASEMYTRIP.NS","POLICYBZR.NS",
    "IXIGO.NS","SAPPHIRE.NS","VEDANT.NS","KALYANKJIL.NS","SENCO.NS",
    "THANGAMAYL.NS","RAJESHEXPO.NS","VAIBHAVGBL.NS","DEVYANI.NS",
    "JUBLFOOD.NS","WESTLIFE.NS","RRKABEL.NS","HONASA.NS","BIKAJI.NS",
    # Pharma / Healthcare Midcap
    "GRANULES.NS","GLENMARK.NS","NATCOPHARM.NS","IPCALAB.NS",
    "MAXHEALTH.NS","KIMS.NS","FORTIS.NS","METROPOLIS.NS",
    "LALPATHLAB.NS","POLYMED.NS","SUVENPHAR.NS",
    # Capital Goods / Defence / Infra
    "KAYNES.NS","TECHNOE.NS","AEROFLEX.NS","ELGIEQUIP.NS","TITAGARH.NS",
    "IRFC.NS","IREDA.NS","NBCC.NS","ENGINERSIN.NS","RITES.NS","NCC.NS",
    "KEC.NS","KALPATPOWR.NS","SJVN.NS","NHPC.NS","THERMAX.NS","JYOTICNC.NS",
    # Chemicals / Specialty
    "DEEPAKNTR.NS","NAVINFLUOR.NS","EPIGRAL.NS","PCBL.NS",
    "SUMICHEM.NS","TATACHEM.NS","GNFC.NS","ATUL.NS",
    "FINEORG.NS","ROSSARI.NS","VINATI.NS","SUDARSCHEM.NS",
    # Auto Ancillaries
    "MOTHERSON.NS","SUNDRMFAST.NS","ENDURANCE.NS","CRAFTSMAN.NS",
    "SUPRAJIT.NS","GABRIEL.NS","EXIDEIND.NS","AMARAJABAT.NS",
    "CEATLTD.NS","BALKRISHNA.NS","LUMAXTECH.NS",
    # Real Estate
    "GODREJPROP.NS","OBEROIRLTY.NS","PHOENIXLTD.NS","IBREALEST.NS",
    "SOBHA.NS","PRESTIGE.NS","BRIGADE.NS","KOLTEPATIL.NS",
    # Cement / Building
    "JKCEMENT.NS","RAMCOCEM.NS","STARCEMENT.NS","KNRCON.NS",
    "VGUARD.NS","FINOLEX.NS","PRINCEPIPE.NS","SKIPPER.NS",
    # FMCG Midcap
    "EMAMILTD.NS","GODFRYPHLP.NS","RADICO.NS","VSTIND.NS",
    "ZYDUSWELL.NS","JYOTHYLAB.NS","BAJAJCON.NS",
    # Logistics
    "BLUEDART.NS","TCI.NS","VRLLOG.NS","ALLCARGO.NS","CONCOR.NS",
    # Textiles
    "TRIDENT.NS","WELSPUNIND.NS","VARDHMAN.NS","RAYMOND.NS","KITEX.NS",
    # PSU / Metals
    "NMDC.NS","NATIONALUM.NS","OIL.NS","MRPL.NS","CHENNPETRO.NS",
    # Specialty / Quality Midcaps
    "SAFARI.NS","AIAENG.NS","GRINDWELL.NS","CARBORUNIV.NS","SKFINDIA.NS",
    "TIMKEN.NS","SCHAEFFLER.NS","CUMMINSIND.NS","LINDEINDIA.NS",
    "BLUESTAR.NS","SYMPHONY.NS","HAWKINCOOK.NS","BAJAJELEC.NS","PVRINOX.NS",
]

# Scan universe presets (deduped)
def _universe(extra):
    seen=set(NIFTY_200); out=NIFTY_200[:]
    for t in extra:
        if t not in seen: out.append(t); seen.add(t)
    return out

UNIVERSE_PRESETS = {
    "⚡ Nifty 50  (~30s)":       NIFTY_200[:50],
    "📊 Nifty 200  (~2 min)":    NIFTY_200,
    "🌐 Nifty 500  (~6 min)":    _universe(NIFTY_500_EXT),
}


# ── Search any NSE stock by company name ─────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def search_nse_stocks(query: str) -> list:
    """
    yfinance Search — finds ANY listed stock by name.
    Works for Kaynes Technology, Nykaa, Delhivery, obscure smallcaps — anything.
    Returns [{ticker, name, sector}]
    """
    results = []
    try:
        # Try with NSE hint first, then bare query
        for q_str in [query + " NSE India", query]:
            s = yf.Search(q_str, max_results=10)
            for q in (s.quotes or []):
                sym = (q.get("symbol") or "").strip()
                exch = q.get("exchange", "")
                if q.get("quoteType") != "EQUITY" or not sym: continue
                if exch in ("NSI", "NSE") or sym.endswith(".NS"):
                    ticker = sym if sym.endswith(".NS") else f"{sym}.NS"
                elif exch in ("BSE", "BOM"):
                    ticker = sym if sym.endswith(".BO") else f"{sym}.BO"
                else:
                    continue
                if any(r["ticker"] == ticker for r in results): continue
                results.append({
                    "ticker": ticker,
                    "name": q.get("shortname") or q.get("longname") or sym,
                    "sector": q.get("sector") or "—",
                    "exchange": "NSE" if ".NS" in ticker else "BSE",
                })
            if results: break
    except Exception:
        pass
    return results[:6]


INDICES = {"^NSEI":"Nifty 50","^NSEBANK":"Bank Nifty","^CNXIT":"IT Index",
           "^CNXMIDCAP":"Midcap 100","^CNXSC":"Smallcap","^INDIAVIX":"India VIX"}

SECTOR_PROXIES = {
    "Technology":["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS"],
    "Banking":["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS"],
    "FMCG":["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS"],
    "Pharma":["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS"],
    "Capital Goods":["LT.NS","SIEMENS.NS","ABB.NS","HAL.NS"],
    "Auto":["MARUTI.NS","M&M.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS"],
    "Energy":["RELIANCE.NS","ONGC.NS","BPCL.NS","NTPC.NS"],
    "Metals":["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS"],
}

SECTOR_PEERS = {
    "Technology":["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS","MPHASIS.NS"],
    "Financial Services":["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","AXISBANK.NS","BAJFINANCE.NS","CHOLAFIN.NS","MUTHOOTFIN.NS"],
    "Consumer Defensive":["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","COLPAL.NS","TATACONSUM.NS"],
    "Healthcare":["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","LUPIN.NS","TORNTPHARM.NS","ALKEM.NS","ABBOTINDIA.NS"],
    "Industrials":["LT.NS","SIEMENS.NS","ABB.NS","HAL.NS","BEL.NS","HAVELLS.NS","CGPOWER.NS","POLYCAB.NS"],
    "Consumer Cyclical":["MARUTI.NS","M&M.NS","TATAMOTORS.NS","EICHERMOT.NS","TITAN.NS","DMART.NS","TRENT.NS","BAJAJ-AUTO.NS"],
    "Energy":["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS","NTPC.NS","POWERGRID.NS"],
    "Basic Materials":["ASIANPAINT.NS","PIDILITIND.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","ASTRAL.NS"],
    "Communication Services":["BHARTIARTL.NS","ZOMATO.NS","NAUKRI.NS"],
}

LENSES = {
    "🏆 Quality":{
        "title":"Quality Compounders","desc":"Sector-aware quality metrics",
        "detail":"Sector-aware quality screen. Banks use ROE/ROA/PB proxies; IT uses margins/FCF/low debt; industrials use ROCE/debt/working-capital proxies available in this demo.",
        "color":"#3b82f6","filters":{"roe_min":15,"de_max":100,"op_margin_min":12},
        "show_cols":["roe","op_margin","de_ratio","pe_ttm"],"labels":["ROE %","Op Margin %","D/E","P/E"],
    },
    "💰 Value":{
        "title":"Deep Value","desc":"Sector-aware valuation discipline",
        "detail":"Stocks trading below intrinsic value. Graham's timeless approach adapted for Dalal Street. The margin of safety is your edge against uncertainty.",
        "color":"#f59e0b","filters":{"pe_max":20,"pb_max":3.0},
        "show_cols":["pe_ttm","pb","graham_mos","div_yield"],"labels":["P/E","P/B","Graham MOS %","Div Yield %"],
    },
    "🚀 Growth":{
        "title":"Growth Rockets","desc":"Sector-aware growth signals",
        "detail":"Accelerating into their potential. Revenue growing >15% and earnings >10% signal expanding markets. PEG matters more than P/E here.",
        "color":"#10b981","filters":{"rev_growth_min":15,"earn_growth_min":10},
        "show_cols":["rev_growth_ttm","earn_growth_ttm","pe_ttm","pe_fwd"],"labels":["Rev Growth %","Earn Growth %","P/E TTM","Fwd P/E"],
    },
    "🛡️ Defensive":{
        "title":"Sleep-Well Stocks","desc":"Sector-aware defensive signals",
        "detail":"Lower volatility, steady dividends, low market sensitivity. Falls less in downturns. Ideal for SIP investors wanting predictable wealth creation.",
        "color":"#8b5cf6","filters":{"div_yield_min":1.5,"beta_max":1.0,"de_max":80},
        "show_cols":["div_yield","beta","de_ratio","current_ratio"],"labels":["Div Yield %","Beta","D/E","Curr. Ratio"],
    },
    "⚙️ ROCE":{
        "title":"Capital Efficient","desc":"Capital efficiency by sector",
        "detail":"Every rupee of capital generating superior returns. ROCE >18% means real economic value creation. Asset-light models scale without equity dilution.",
        "color":"#06b6d4","filters":{"roce_min":18,"op_margin_min":12},
        "show_cols":["roce","op_margin","fcf_yield","pe_ttm"],"labels":["ROCE %","Op Margin %","FCF Yield %","P/E"],
    },
    "🏛️ Governance":{
        "title":"Governance Proxy","desc":"Higher CG-Proxy · Fewer visible risk signals",
        "detail":"A preliminary governance-risk proxy using only available yfinance fields. It is useful for triage, but it is not a formal corporate-governance rating. Verify with exchange filings, auditor notes, promoter pledges, related-party transactions and SEBI disclosures before acting.",
        "color":"#ec4899","filters":{"cg_score_min":60},
        "show_cols":["cg_score","roe","de_ratio","div_yield"],"labels":["CG-Proxy","ROE %","D/E","Div Yield %"],
    },
    "📡 Consensus":{
        "title":"Analyst Conviction","desc":"Strong buy ratings · >20% upside targets",
        "detail":"Professional consensus sees significant upside. 5+ analysts covering with >20% average upside. Verify independently—consensus is wrong at turning points.",
        "color":"#f97316","filters":{"analyst_upside_min":20,"analyst_count_min":5},
        "show_cols":["analyst_upside","pe_fwd","rev_growth_ttm","from_52w_high"],"labels":["Upside %","Fwd P/E","Rev Growth %","vs 52W High %"],
    },
}

COLORS=["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899","#a78bfa","#34d399"]
EPS=1e-12; ANN=252

# ── Session State ────────────────────────────────────────────────────────────
for k,v in {
    "page":"pulse","active_lens":None,"lens_cache":{},
    "analyze_ticker":"TCS.NS","portfolio_tickers":[],
    "portfolio_data":None,"portfolio_opt":None,
    "api_key":"","api_provider":"Groq",
    "radar_results":None,"radar_ts":None,
    "lens_universe_sel":"📊 Nifty 200  (~2 min)",
    # Persist Natural Language screener results so row action buttons work
    # after Streamlit reruns. Buttons inside result rows only exist if the
    # result dataframe is re-rendered on the next run.
    "nl_results_df":None,"nl_results_title":None,
    "nl_results_query":None,"nl_results_universe":None,
}.items():
    if k not in st.session_state: st.session_state[k]=v


# ╔══════════════════════════════════════════════════════════════════╗
#  UTILITY FUNCTIONS
# ╚══════════════════════════════════════════════════════════════════╝
def _r(v,nd=2):
    try:
        f=float(v); return None if (np.isnan(f) or np.isinf(f)) else round(f,nd)
    except: return None

def _p(v):
    try:
        f=float(v); return None if (np.isnan(f) or np.isinf(f)) else round(f*100,2)
    except: return None

def _cap(v):
    if v is None: return "—"
    v=float(v)
    if v>=1e12: return f"₹{v/1e12:.1f}T"
    if v>=1e9:  return f"₹{v/1e9:.1f}B"
    if v>=1e7:  return f"₹{v/1e7:.0f}Cr"
    return f"₹{v/1e6:.1f}M"

def _dark(fig,h=360,title=""):
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#050710",
        height=h,title=title,title_font=dict(size=12,color="#3d4466"),
        margin=dict(l=0,r=0,t=36 if title else 12,b=10),
        xaxis=dict(showgrid=True,gridcolor="#0c0e1c",tickfont=dict(size=10,color="#3d4466")),
        yaxis=dict(showgrid=True,gridcolor="#0c0e1c",tickfont=dict(size=10,color="#3d4466")),
        legend=dict(font=dict(size=11,color="#6b7280"),bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def _score_color(s):
    if s is None: return "#3d4466"
    if s>=75: return "#22c55e"
    if s>=55: return "#f59e0b"
    if s>=35: return "#f97316"
    return "#ef4444"

def _score_grade(s):
    if s is None: return "N/A"
    if s>=80: return "A"
    if s>=65: return "B"
    if s>=50: return "C"
    if s>=35: return "D"
    return "F"


def _is_financial_sector(sector="", industry=""):
    """Banks/NBFCs/insurers need different interpretation for D/E, liquidity, and margins."""
    text=f"{sector or ''} {industry or ''}".lower()
    return any(x in text for x in [
        "bank", "financial", "finance", "insurance", "credit", "nbfc",
        "capital markets", "asset management", "mortgage"
    ])


def _data_availability_score(m):
    """Percentage of key data fields available for this app's signals."""
    keys=[
        "price","market_cap","pe_ttm","pb","roe","roa","roce","op_margin","net_margin",
        "rev_growth_ttm","earn_growth_ttm","de_ratio","current_ratio","beta","div_yield",
        "year_high","year_low","fcf_yield","analyst_target","analyst_count","cg_score"
    ]
    available=sum(1 for k in keys if m.get(k) is not None)
    return round(available/len(keys)*100,0)


def _data_confidence_label(score):
    if score is None: return "Unknown"
    if score>=80: return "High"
    if score>=60: return "Moderate"
    if score>=40: return "Low"
    return "Very low"


def _fmt_num(v, suffix="", nd=1, prefix=""):
    try:
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        return f"{prefix}{float(v):,.{nd}f}{suffix}"
    except Exception:
        return "—"


# ╔══════════════════════════════════════════════════════════════════╗
#  SECTOR-AWARE SCORING ENGINE
# ╚══════════════════════════════════════════════════════════════════╝
SECTOR_MODEL_NOTES = {
    "FINANCIAL": {
        "name": "Financials: Banks/NBFCs/Insurance",
        "logic": "Uses ROE, ROA, P/B, growth, beta, dividend, CG-Proxy and data availability. It deliberately avoids treating D/E/current ratio like a manufacturing company.",
        "missing": "Future API inputs: GNPA, NNPA, NIM, CASA, credit cost, CRAR, PCR, AUM growth, ALM gaps and solvency ratio for insurers.",
    },
    "TECH": {
        "name": "IT / Digital / Software",
        "logic": "Emphasizes operating margin, ROE, FCF yield, growth, low debt, earnings quality and valuation discipline.",
        "missing": "Future API inputs: deal wins, attrition, client concentration, constant-currency growth and offshore/onshore mix.",
    },
    "HEALTHCARE": {
        "name": "Pharma / Healthcare",
        "logic": "Balances growth, operating margin, ROCE/ROE, leverage, valuation and earnings quality.",
        "missing": "Future API inputs: USFDA observations, product concentration, R&D intensity, ANDA pipeline and hospital occupancy/ARPOB where relevant.",
    },
    "CONSUMER_DEFENSIVE": {
        "name": "FMCG / Consumer Defensive",
        "logic": "Prioritizes ROE, margins, moderate growth, dividend consistency, beta, leverage and valuation reasonableness.",
        "missing": "Future API inputs: volume growth, rural/urban split, ad-spend intensity, distribution reach and raw-material inflation sensitivity.",
    },
    "CONSUMER_CYCLICAL": {
        "name": "Consumer Cyclical / Auto / Retail / Travel",
        "logic": "Rewards growth and capital efficiency but penalizes high leverage, very high beta and weak margins more than defensive sectors.",
        "missing": "Future API inputs: same-store sales, order book, inventory days, financing penetration and demand-cycle indicators.",
    },
    "INDUSTRIALS": {
        "name": "Industrials / Capital Goods / Infra",
        "logic": "Emphasizes ROCE, operating margin, debt control, current ratio, revenue growth, FCF yield and CG-Proxy.",
        "missing": "Future API inputs: order book, execution cycle, working-capital days, client concentration and government/payment receivable risk.",
    },
    "ENERGY_MATERIALS": {
        "name": "Energy / Materials / Metals / Chemicals",
        "logic": "Treats cyclicals differently: more weight to balance-sheet strength, FCF yield, dividend, P/B and cycle-aware valuation than pure growth.",
        "missing": "Future API inputs: commodity spreads, utilization, volume growth, reserves, regulated returns and input-cost sensitivity.",
    },
    "COMMUNICATION_NEWAGE": {
        "name": "Communication / Internet / New-age",
        "logic": "Allows lower current profitability if revenue growth, gross margin, balance-sheet liquidity and price momentum are acceptable.",
        "missing": "Future API inputs: unit economics, contribution margin, customer acquisition cost, retention/cohort data and cash runway.",
    },
    "GENERAL": {
        "name": "General Non-financial",
        "logic": "Balanced model using ROE/ROCE, margin, growth, leverage, FCF, valuation, CG-Proxy and data availability.",
        "missing": "Future API inputs depend on the exact industry.",
    },
}


def _sector_model_key(sector="", industry="", name=""):
    """Classify stocks into practical score models using yfinance sector/industry text."""
    text=f"{sector or ''} {industry or ''} {name or ''}".lower()
    if any(x in text for x in ["bank", "financial", "finance", "insurance", "credit", "nbfc", "capital markets", "asset management", "mortgage", "housing finance"]):
        return "FINANCIAL"
    if any(x in text for x in ["software", "information technology", "it services", "semiconductor", "computer", "digital", "technology", "internet content", "data processing"]):
        return "TECH"
    if any(x in text for x in ["pharma", "biotechnology", "drug", "healthcare", "hospital", "diagnostic", "medical", "life sciences"]):
        return "HEALTHCARE"
    if any(x in text for x in ["consumer defensive", "fmcg", "household", "personal products", "packaged foods", "beverages", "tobacco", "food distribution"]):
        return "CONSUMER_DEFENSIVE"
    if any(x in text for x in ["auto", "automobile", "consumer cyclical", "retail", "restaurant", "apparel", "luxury", "travel", "leisure", "hotel", "e-commerce", "jewellery", "jewelry"]):
        return "CONSUMER_CYCLICAL"
    if any(x in text for x in ["industrial", "capital goods", "engineering", "infrastructure", "construction", "defense", "defence", "electrical", "machinery", "rail", "ship", "aerospace", "utilities"]):
        return "INDUSTRIALS"
    if any(x in text for x in ["energy", "oil", "gas", "coal", "power", "metal", "steel", "aluminum", "aluminium", "copper", "mining", "chemical", "materials", "cement", "fertilizer"]):
        return "ENERGY_MATERIALS"
    if any(x in text for x in ["communication", "telecom", "media", "entertainment", "interactive media", "internet retail", "new age"]):
        return "COMMUNICATION_NEWAGE"
    return "GENERAL"


def _clamp(x, lo=0, hi=100):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def _score_high(v, poor, good):
    """Higher is better: 0 at poor, 100 at good."""
    if v is None: return None
    return _clamp((float(v)-poor)/(good-poor+EPS)*100)


def _score_low(v, good, poor):
    """Lower is better: 100 at good or below, 0 at poor or above."""
    if v is None: return None
    return _clamp((poor-float(v))/(poor-good+EPS)*100)


def _score_range(v, ideal_low, ideal_high, bad_low, bad_high):
    """Best inside ideal range; penalizes both too low and too high."""
    if v is None: return None
    v=float(v)
    if v<=bad_low or v>=bad_high: return 0
    if ideal_low<=v<=ideal_high: return 100
    if v<ideal_low: return _clamp((v-bad_low)/(ideal_low-bad_low+EPS)*100)
    return _clamp((bad_high-v)/(bad_high-ideal_high+EPS)*100)


def _component(label, value, weight, mode, poor=None, good=None, ideal_low=None, ideal_high=None, bad_low=None, bad_high=None, note=""):
    if mode=="high": score=_score_high(value, poor, good)
    elif mode=="low": score=_score_low(value, good, poor)
    elif mode=="range": score=_score_range(value, ideal_low, ideal_high, bad_low, bad_high)
    elif mode=="direct": score=_clamp(value)
    else: score=None
    if score is None: return None
    return {"label":label,"value":value,"score":score,"weight":weight,"note":note}


def _weighted_total(components):
    usable=[c for c in components if c]
    w=sum(c["weight"] for c in usable)
    if w<=0:
        return {"score":None,"coverage":0,"components":[],"grade":"N/A","color":"#3d4466"}
    score=sum(c["score"]*c["weight"] for c in usable)/w
    coverage=min(100, round(w,0))
    return {"score":round(score,1),"coverage":coverage,"components":usable,"grade":_score_grade(score),"color":_score_color(score)}


def _lens_key(lens_name):
    s=(lens_name or "").lower()
    if "quality" in s: return "QUALITY"
    if "value" in s: return "VALUE"
    if "growth" in s: return "GROWTH"
    if "defensive" in s: return "DEFENSIVE"
    if "roce" in s: return "CAPITAL_EFFICIENCY"
    if "governance" in s: return "GOVERNANCE"
    if "consensus" in s: return "CONSENSUS"
    return "ALPHA"


def compute_sector_alpha_score(m, lens_name=None):
    """Sector-aware 0-100 score. Uses only available yfinance demo fields and exposes coverage."""
    model=_sector_model_key(m.get("sector"), m.get("industry"), m.get("name"))
    lens=_lens_key(lens_name)
    comps=[]
    add=lambda *a, **k: comps.append(_component(*a, **k))

    # Shared governance/data components. These are low-weight modifiers, not a substitute for filings.
    def add_governance_data(cg_w=8, dq_w=5):
        add("CG-Proxy", m.get("cg_score"), cg_w, "direct", note="yfinance governance/ownership proxy")
        add("Data availability", m.get("data_quality"), dq_w, "direct", note="how much required demo data was available")

    if lens=="GOVERNANCE":
        add("CG-Proxy", m.get("cg_score"), 45, "direct")
        add("Insider/management holding proxy", m.get("insider_pct"), 20, "high", poor=5, good=45)
        add("Dividend discipline", m.get("div_yield"), 10, "high", poor=0, good=2.5)
        add("Cash quality", m.get("fcf_yield"), 15, "high", poor=-2, good=5)
        add("Data availability", m.get("data_quality"), 10, "direct")
    elif lens=="CONSENSUS":
        add("Analyst upside", m.get("analyst_upside"), 35, "high", poor=0, good=35)
        add("Analyst coverage", m.get("analyst_count"), 20, "high", poor=1, good=12)
        add("Forward P/E reasonableness", m.get("pe_fwd"), 15, "range", ideal_low=8, ideal_high=28, bad_low=0, bad_high=80)
        add("Growth support", m.get("rev_growth_ttm"), 15, "high", poor=-5, good=25)
        add_governance_data(10,5)
    elif model=="FINANCIAL":
        if lens=="VALUE":
            add("P/B valuation", m.get("pb"), 30, "range", ideal_low=0.7, ideal_high=2.2, bad_low=0, bad_high=6)
            add("P/E valuation", m.get("pe_ttm"), 20, "range", ideal_low=6, ideal_high=18, bad_low=0, bad_high=45)
            add("ROE support", m.get("roe"), 15, "high", poor=6, good=17)
            add("Dividend yield", m.get("div_yield"), 10, "high", poor=0, good=3)
            add_governance_data(15,10)
        elif lens=="GROWTH":
            add("Earnings growth", m.get("earn_growth_ttm"), 30, "high", poor=-5, good=25)
            add("Revenue/interest-income growth proxy", m.get("rev_growth_ttm"), 25, "high", poor=0, good=20)
            add("ROE", m.get("roe"), 15, "high", poor=6, good=17)
            add("ROA", m.get("roa"), 15, "high", poor=0.3, good=1.5)
            add_governance_data(10,5)
        elif lens=="DEFENSIVE":
            add("Beta", m.get("beta"), 20, "low", good=0.7, poor=1.6)
            add("ROA", m.get("roa"), 20, "high", poor=0.3, good=1.4)
            add("ROE", m.get("roe"), 20, "high", poor=6, good=16)
            add("Dividend yield", m.get("div_yield"), 15, "high", poor=0, good=3)
            add("Drawdown from 52W high", abs(m.get("from_52w_high") or 0), 10, "low", good=5, poor=45)
            add_governance_data(10,5)
        elif lens=="CAPITAL_EFFICIENCY":
            add("ROE", m.get("roe"), 35, "high", poor=6, good=18)
            add("ROA", m.get("roa"), 35, "high", poor=0.3, good=1.6)
            add("P/B discipline", m.get("pb"), 10, "range", ideal_low=0.8, ideal_high=3.0, bad_low=0, bad_high=7)
            add_governance_data(15,5)
        else:  # QUALITY / ALPHA
            add("ROE", m.get("roe"), 25, "high", poor=6, good=18)
            add("ROA", m.get("roa"), 25, "high", poor=0.3, good=1.6)
            add("Earnings growth", m.get("earn_growth_ttm"), 15, "high", poor=-5, good=22)
            add("P/B discipline", m.get("pb"), 10, "range", ideal_low=0.8, ideal_high=3.5, bad_low=0, bad_high=8)
            add("Beta", m.get("beta"), 8, "low", good=0.8, poor=1.8)
            add_governance_data(12,5)
    elif model=="TECH":
        if lens=="VALUE":
            add("P/E valuation", m.get("pe_ttm"), 25, "range", ideal_low=12, ideal_high=30, bad_low=0, bad_high=90)
            add("FCF yield", m.get("fcf_yield"), 25, "high", poor=-2, good=5)
            add("ROE support", m.get("roe"), 15, "high", poor=8, good=25)
            add("Growth support", m.get("rev_growth_ttm"), 15, "high", poor=-5, good=20)
            add_governance_data(15,5)
        elif lens=="GROWTH":
            add("Revenue growth", m.get("rev_growth_ttm"), 30, "high", poor=0, good=25)
            add("Earnings growth", m.get("earn_growth_ttm"), 25, "high", poor=-5, good=25)
            add("Operating margin", m.get("op_margin"), 20, "high", poor=8, good=28)
            add("FCF yield", m.get("fcf_yield"), 10, "high", poor=-2, good=5)
            add_governance_data(10,5)
        elif lens=="DEFENSIVE":
            add("Operating margin", m.get("op_margin"), 25, "high", poor=8, good=30)
            add("Low debt", m.get("de_ratio"), 20, "low", good=0, poor=80)
            add("Beta", m.get("beta"), 20, "low", good=0.7, poor=1.6)
            add("FCF yield", m.get("fcf_yield"), 15, "high", poor=-2, good=5)
            add_governance_data(15,5)
        else:
            add("Operating margin", m.get("op_margin"), 25, "high", poor=8, good=30)
            add("ROE", m.get("roe"), 20, "high", poor=8, good=25)
            add("Revenue growth", m.get("rev_growth_ttm"), 15, "high", poor=-5, good=22)
            add("FCF yield", m.get("fcf_yield"), 15, "high", poor=-2, good=5)
            add("Low debt", m.get("de_ratio"), 10, "low", good=0, poor=100)
            add_governance_data(10,5)
    elif model=="HEALTHCARE":
        if lens=="VALUE":
            add("P/E valuation", m.get("pe_ttm"), 25, "range", ideal_low=12, ideal_high=28, bad_low=0, bad_high=80)
            add("P/B valuation", m.get("pb"), 10, "range", ideal_low=1, ideal_high=5, bad_low=0, bad_high=14)
            add("ROCE support", m.get("roce"), 20, "high", poor=8, good=22)
            add("FCF yield", m.get("fcf_yield"), 20, "high", poor=-2, good=5)
            add_governance_data(15,10)
        elif lens=="GROWTH":
            add("Revenue growth", m.get("rev_growth_ttm"), 25, "high", poor=-5, good=22)
            add("Earnings growth", m.get("earn_growth_ttm"), 25, "high", poor=-10, good=25)
            add("Operating margin", m.get("op_margin"), 20, "high", poor=8, good=25)
            add("ROCE", m.get("roce"), 15, "high", poor=8, good=22)
            add_governance_data(10,5)
        else:
            add("Operating margin", m.get("op_margin"), 22, "high", poor=8, good=25)
            add("ROCE", m.get("roce"), 22, "high", poor=8, good=24)
            add("ROE", m.get("roe"), 15, "high", poor=6, good=22)
            add("Low debt", m.get("de_ratio"), 15, "low", good=10, poor=120)
            add("Revenue growth", m.get("rev_growth_ttm"), 11, "high", poor=-5, good=18)
            add_governance_data(10,5)
    elif model=="CONSUMER_DEFENSIVE":
        if lens=="DEFENSIVE":
            add("Beta", m.get("beta"), 25, "low", good=0.55, poor=1.3)
            add("Operating margin", m.get("op_margin"), 20, "high", poor=8, good=25)
            add("ROE", m.get("roe"), 20, "high", poor=8, good=25)
            add("Dividend yield", m.get("div_yield"), 15, "high", poor=0, good=2.5)
            add("Low debt", m.get("de_ratio"), 10, "low", good=0, poor=100)
            add_governance_data(5,5)
        elif lens=="VALUE":
            add("P/E valuation", m.get("pe_ttm"), 25, "range", ideal_low=15, ideal_high=35, bad_low=0, bad_high=90)
            add("Dividend yield", m.get("div_yield"), 20, "high", poor=0, good=2.5)
            add("ROE support", m.get("roe"), 20, "high", poor=8, good=25)
            add("FCF yield", m.get("fcf_yield"), 15, "high", poor=-2, good=4)
            add_governance_data(15,5)
        else:
            add("ROE", m.get("roe"), 25, "high", poor=8, good=25)
            add("Operating margin", m.get("op_margin"), 22, "high", poor=8, good=25)
            add("Revenue growth", m.get("rev_growth_ttm"), 15, "high", poor=-3, good=15)
            add("Beta", m.get("beta"), 10, "low", good=0.6, poor=1.5)
            add("Low debt", m.get("de_ratio"), 10, "low", good=0, poor=100)
            add_governance_data(13,5)
    elif model=="CONSUMER_CYCLICAL":
        add("Revenue growth", m.get("rev_growth_ttm"), 22 if lens in ["GROWTH","ALPHA"] else 15, "high", poor=-8, good=25)
        add("Earnings growth", m.get("earn_growth_ttm"), 18 if lens=="GROWTH" else 12, "high", poor=-12, good=30)
        add("ROCE", m.get("roce"), 18, "high", poor=6, good=22)
        add("Operating margin", m.get("op_margin"), 15, "high", poor=3, good=18)
        add("Debt control", m.get("de_ratio"), 15, "low", good=20, poor=150)
        add("Beta", m.get("beta"), 5 if lens!="DEFENSIVE" else 15, "low", good=0.8, poor=2.0)
        if lens=="VALUE": add("P/E valuation", m.get("pe_ttm"), 15, "range", ideal_low=8, ideal_high=25, bad_low=0, bad_high=70)
        add_governance_data(10,5)
    elif model=="INDUSTRIALS":
        add("ROCE", m.get("roce"), 25 if lens in ["CAPITAL_EFFICIENCY","QUALITY","ALPHA"] else 18, "high", poor=7, good=24)
        add("Operating margin", m.get("op_margin"), 18, "high", poor=5, good=18)
        add("Debt control", m.get("de_ratio"), 18, "low", good=20, poor=150)
        add("Current ratio", m.get("current_ratio"), 10, "range", ideal_low=1.1, ideal_high=2.5, bad_low=0.5, bad_high=5)
        add("Revenue growth", m.get("rev_growth_ttm"), 14 if lens=="GROWTH" else 10, "high", poor=-5, good=20)
        add("FCF yield", m.get("fcf_yield"), 10, "high", poor=-3, good=5)
        if lens=="VALUE": add("P/E valuation", m.get("pe_ttm"), 15, "range", ideal_low=8, ideal_high=25, bad_low=0, bad_high=70)
        add_governance_data(10,5)
    elif model=="ENERGY_MATERIALS":
        add("Balance-sheet debt control", m.get("de_ratio"), 20, "low", good=25, poor=160)
        add("ROCE", m.get("roce"), 18, "high", poor=5, good=20)
        add("FCF yield", m.get("fcf_yield"), 18, "high", poor=-3, good=7)
        add("Dividend yield", m.get("div_yield"), 15, "high", poor=0, good=4)
        add("P/B cycle-aware valuation", m.get("pb"), 12, "range", ideal_low=0.5, ideal_high=2.5, bad_low=0, bad_high=8)
        add("Earnings growth", m.get("earn_growth_ttm"), 7 if lens=="GROWTH" else 5, "high", poor=-20, good=25)
        add_governance_data(10,5)
    elif model=="COMMUNICATION_NEWAGE":
        add("Revenue growth", m.get("rev_growth_ttm"), 28, "high", poor=-5, good=30)
        add("Gross margin", m.get("gross_margin"), 15, "high", poor=20, good=65)
        add("Net margin progress", m.get("net_margin"), 15, "high", poor=-30, good=15)
        add("Current ratio / liquidity proxy", m.get("current_ratio"), 10, "range", ideal_low=1.0, ideal_high=4.0, bad_low=0.4, bad_high=10)
        add("52W damage control", abs(m.get("from_52w_high") or 0), 10, "low", good=5, poor=60)
        add("Beta", m.get("beta"), 7, "low", good=0.8, poor=2.2)
        add_governance_data(10,5)
    else:
        if lens=="VALUE":
            add("P/E valuation", m.get("pe_ttm"), 25, "range", ideal_low=8, ideal_high=22, bad_low=0, bad_high=70)
            add("P/B valuation", m.get("pb"), 15, "range", ideal_low=0.7, ideal_high=3.5, bad_low=0, bad_high=10)
            add("FCF yield", m.get("fcf_yield"), 20, "high", poor=-2, good=5)
            add("ROE support", m.get("roe"), 15, "high", poor=5, good=20)
            add_governance_data(15,10)
        elif lens=="GROWTH":
            add("Revenue growth", m.get("rev_growth_ttm"), 30, "high", poor=-5, good=25)
            add("Earnings growth", m.get("earn_growth_ttm"), 25, "high", poor=-10, good=30)
            add("Operating margin", m.get("op_margin"), 15, "high", poor=5, good=20)
            add("ROE", m.get("roe"), 15, "high", poor=5, good=20)
            add_governance_data(10,5)
        elif lens=="DEFENSIVE":
            add("Beta", m.get("beta"), 25, "low", good=0.7, poor=1.7)
            add("Dividend yield", m.get("div_yield"), 20, "high", poor=0, good=3)
            add("Debt control", m.get("de_ratio"), 20, "low", good=20, poor=140)
            add("ROE", m.get("roe"), 15, "high", poor=5, good=18)
            add_governance_data(15,5)
        else:
            add("ROCE", m.get("roce"), 20, "high", poor=6, good=22)
            add("ROE", m.get("roe"), 15, "high", poor=5, good=20)
            add("Operating margin", m.get("op_margin"), 15, "high", poor=5, good=20)
            add("Revenue growth", m.get("rev_growth_ttm"), 15, "high", poor=-5, good=20)
            add("Debt control", m.get("de_ratio"), 15, "low", good=20, poor=140)
            add("FCF yield", m.get("fcf_yield"), 10, "high", poor=-2, good=5)
            add_governance_data(5,5)

    result=_weighted_total(comps)
    result["model_key"]=model
    result["model_name"]=SECTOR_MODEL_NOTES[model]["name"]
    result["model_logic"]=SECTOR_MODEL_NOTES[model]["logic"]
    result["missing_true_metrics"]=SECTOR_MODEL_NOTES[model]["missing"]
    result["lens"]=_lens_key(lens_name)
    return result


def passes_sector_lens(m, lens_name, custom_filters=None):
    """Sector-aware lens pre-filter. Custom NLP filters are applied literally; curated lenses use sector-specific gates."""
    if custom_filters:
        field_map={"roe_min":"roe","de_max":"de_ratio","op_margin_min":"op_margin",
                   "pe_max":"pe_ttm","pe_min":"pe_ttm","pb_max":"pb",
                   "rev_growth_min":"rev_growth_ttm","earn_growth_min":"earn_growth_ttm",
                   "div_yield_min":"div_yield","beta_max":"beta","roce_min":"roce",
                   "analyst_upside_min":"analyst_upside","analyst_count_min":"analyst_count",
                   "cg_score_min":"cg_score","fcf_yield_min":"fcf_yield","roa_min":"roa"}
        for fk,fv in custom_filters.items():
            if fk=="sector":
                if str(fv).lower() not in (m.get("sector","") or "").lower(): return False
                continue
            field=field_map.get(fk)
            if not field: continue
            val=m.get(field)
            if val is None: return False
            if fk.endswith("_min") and val<fv: return False
            if fk.endswith("_max") and val>fv: return False
            if fk=="pe_max" and val<=0: return False
        return True

    model=_sector_model_key(m.get("sector"),m.get("industry"),m.get("name"))
    lens=_lens_key(lens_name)
    dq=m.get("data_quality") or 0
    if dq<30: return False
    pe=m.get("pe_ttm"); pb=m.get("pb"); roe=m.get("roe"); roa=m.get("roa"); roce=m.get("roce")
    rev=m.get("rev_growth_ttm"); earn=m.get("earn_growth_ttm"); debt=m.get("de_ratio"); op=m.get("op_margin")
    beta=m.get("beta"); div=m.get("div_yield"); cg=m.get("cg_score")

    if lens=="QUALITY":
        if model=="FINANCIAL": return ((roe is not None and roe>=9) or (roa is not None and roa>=0.6)) and (cg is None or cg>=35)
        if model=="TECH": return (roe is not None and roe>=12) and (op is not None and op>=10) and (debt is None or debt<=100)
        if model=="CONSUMER_DEFENSIVE": return (roe is not None and roe>=12) and (op is not None and op>=8)
        return ((roce is not None and roce>=10) or (roe is not None and roe>=10)) and (debt is None or debt<=160)
    if lens=="VALUE":
        if model=="FINANCIAL": return ((pb is not None and pb<=3.8) or (pe is not None and 0<pe<=22)) and (roe is None or roe>=6)
        return ((pe is not None and 0<pe<=32) or (pb is not None and pb<=4.5) or (m.get("fcf_yield") is not None and m.get("fcf_yield")>=3) or (m.get("graham_mos") is not None and m.get("graham_mos")>0))
    if lens=="GROWTH":
        return ((rev is not None and rev>=10) or (earn is not None and earn>=10)) and (cg is None or cg>=30)
    if lens=="DEFENSIVE":
        if model=="FINANCIAL": return (beta is None or beta<=1.25) and ((div is not None and div>=0.8) or (roa is not None and roa>=0.7))
        return (beta is None or beta<=1.25) and (debt is None or debt<=130) and ((div is not None and div>=0.8) or (roe is not None and roe>=10))
    if lens=="CAPITAL_EFFICIENCY":
        if model=="FINANCIAL": return ((roe is not None and roe>=11) or (roa is not None and roa>=0.8))
        return ((roce is not None and roce>=14) or (roe is not None and roe>=15))
    if lens=="GOVERNANCE":
        return cg is not None and cg>=50
    if lens=="CONSENSUS":
        return (m.get("analyst_upside") is not None and m.get("analyst_upside")>=15) and (m.get("analyst_count") is not None and m.get("analyst_count")>=3)
    return True


# ╔══════════════════════════════════════════════════════════════════╗
#  TECHNICAL ANALYSIS ENGINE
# ╚══════════════════════════════════════════════════════════════════╝
def compute_rsi(prices,window=14):
    delta=prices.diff()
    gain=delta.where(delta>0,0.0).rolling(window).mean()
    loss=-delta.where(delta<0,0.0).rolling(window).mean()
    rs=gain/(loss+EPS)
    return 100-(100/(1+rs))

def compute_macd(prices,fast=12,slow=26,signal=9):
    ef=prices.ewm(span=fast,adjust=False).mean()
    es=prices.ewm(span=slow,adjust=False).mean()
    macd=ef-es; sig=macd.ewm(span=signal,adjust=False).mean()
    return macd,sig,macd-sig

def compute_bb(prices,window=20,num_std=2):
    sma=prices.rolling(window).mean(); std=prices.rolling(window).std()
    return sma+num_std*std,sma,sma-num_std*std

def get_technical_signals(prices):
    """Returns dict of current technical signal values and interpretations."""
    if len(prices)<50:
        return {"error":"Need at least 50 price points"}
    out={}
    # RSI
    rsi=compute_rsi(prices)
    out["rsi"]=round(float(rsi.iloc[-1]),1)
    if out["rsi"]<30: out["rsi_sig"]="Oversold 🟢"; out["rsi_col"]="#22c55e"
    elif out["rsi"]>70: out["rsi_sig"]="Overbought 🔴"; out["rsi_col"]="#ef4444"
    else: out["rsi_sig"]="Neutral ⬜"; out["rsi_col"]="#6b7280"
    # MACD
    macd,sig,hist=compute_macd(prices)
    out["macd"]=round(float(macd.iloc[-1]),2)
    out["macd_sig_val"]=round(float(sig.iloc[-1]),2)
    out["macd_hist"]=round(float(hist.iloc[-1]),2)
    if hist.iloc[-1]>0 and hist.iloc[-2]<0: out["macd_sig"]="Bullish Crossover 🟢"; out["macd_col"]="#22c55e"
    elif hist.iloc[-1]<0 and hist.iloc[-2]>0: out["macd_sig"]="Bearish Crossover 🔴"; out["macd_col"]="#ef4444"
    elif hist.iloc[-1]>0: out["macd_sig"]="Bullish ↑"; out["macd_col"]="#86efac"
    else: out["macd_sig"]="Bearish ↓"; out["macd_col"]="#fca5a5"
    # SMA
    if len(prices)>=200:
        sma50=float(prices.rolling(50).mean().iloc[-1])
        sma200=float(prices.rolling(200).mean().iloc[-1])
        p=float(prices.iloc[-1])
        out["sma50"]=round(sma50,2); out["sma200"]=round(sma200,2)
        if p>sma50>sma200: out["sma_sig"]="Strong Uptrend 🟢"; out["sma_col"]="#22c55e"
        elif p>sma50: out["sma_sig"]="Above 50 SMA ↑"; out["sma_col"]="#86efac"
        elif p<sma50<sma200: out["sma_sig"]="Strong Downtrend 🔴"; out["sma_col"]="#ef4444"
        else: out["sma_sig"]="Mixed ⬜"; out["sma_col"]="#6b7280"
    # Bollinger
    bb_up,bb_mid,bb_lo=compute_bb(prices)
    p=float(prices.iloc[-1]); bup=float(bb_up.iloc[-1]); blo=float(bb_lo.iloc[-1]); bwid=(bup-blo)
    out["bb_pct"]=round((p-blo)/(bwid+EPS)*100,1)
    if out["bb_pct"]>90: out["bb_sig"]="Near Upper Band 🔴"; out["bb_col"]="#ef4444"
    elif out["bb_pct"]<10: out["bb_sig"]="Near Lower Band 🟢"; out["bb_col"]="#22c55e"
    else: out["bb_sig"]=f"Mid Range ({out['bb_pct']:.0f}%)"; out["bb_col"]="#6b7280"
    # 52W momentum
    if len(prices)>=252:
        out["mom_52w"]=round((float(prices.iloc[-1])/float(prices.iloc[-252])-1)*100,1)
    # Trend composite score
    s=50
    if out.get("rsi",50)<30: s+=20
    elif out.get("rsi",50)>70: s-=20
    if out.get("macd_hist",0)>0: s+=15
    else: s-=15
    if out.get("bb_pct",50)<20: s+=10
    elif out.get("bb_pct",50)>80: s-=10
    out["tech_score"]=max(0,min(100,s))
    return out


# ╔══════════════════════════════════════════════════════════════════╗
#  CG-SCORE™ — CORPORATE GOVERNANCE SCORE (0-100)
# ╚══════════════════════════════════════════════════════════════════╝
def compute_cg_score(info, t_obj):
    """
    CG-Proxy Score™ (0-100, when enough data exists).
    This is a preliminary governance-risk proxy, not a formal governance rating.

    Available-signal model:
      Signal 1: Yahoo audit/board/compensation/shareholder-rights risk where available (35 weight)
      Signal 2: Insider/management holding where available (25 weight) — not the same as Indian promoter holding
      Signal 3: Dividend presence/track record proxy (15 weight)
      Signal 4: Debt trajectory versus revenue (15 weight)
      Signal 5: Cash conversion CFO/NI (10 weight)

    Score is normalized to available signals and returned with a coverage percentage.
    """
    score_sum=0.0; weight_sum=0.0; details=[]

    # Signal 1: Yahoo governance risk scores
    risks=[info.get(k) for k in ["auditRisk","boardRisk","compensationRisk","shareHolderRightsRisk"]]
    risks=[r for r in risks if r is not None]
    if risks:
        weight=35; avg=sum(risks)/len(risks)
        pts=max(0,(10-avg)/10)*weight
        score_sum+=pts; weight_sum+=weight
        if avg<=3: details.append(("g",f"Low available governance-risk proxy score ({avg:.0f}/10)."))
        elif avg<=6: details.append(("y",f"Moderate available governance-risk proxy score ({avg:.0f}/10)."))
        else: details.append(("r",f"High available governance-risk proxy score ({avg:.0f}/10); verify filings and auditor comments."))
    else:
        details.append(("y","Yahoo governance-risk fields are unavailable; CG-Proxy confidence is reduced."))

    # Signal 2: Insider/management holding proxy
    insider=info.get("heldPercentInsiders")
    if insider is not None:
        weight=25; pct=float(insider)*100
        if pct>=51: pts=25; details.append(("g",f"High insider/management holding proxy ({pct:.1f}%)."))
        elif pct>=30: pts=18; details.append(("g",f"Meaningful insider/management holding proxy ({pct:.1f}%)."))
        elif pct>=15: pts=10; details.append(("y",f"Moderate insider/management holding proxy ({pct:.1f}%)."))
        else: pts=3; details.append(("r",f"Low insider/management holding proxy ({pct:.1f}%). This is not the same as promoter holding."))
        score_sum+=pts; weight_sum+=weight
    else:
        details.append(("y","Insider/management holding proxy is unavailable; do not infer promoter holding from this app."))

    # Signal 3: Dividend track record proxy
    dy=info.get("dividendYield"); fya=info.get("fiveYearAvgDividendYield")
    weight=15
    if dy is not None:
        if float(dy)>0 and fya and float(fya)>0:
            pts=15; details.append(("g",f"Dividend payer with available 5Y dividend-yield history ({float(dy)*100:.1f}% current yield)."))
        elif float(dy)>0:
            pts=8; details.append(("y",f"Pays dividend ({float(dy)*100:.1f}% yield); dividend consistency not fully verified."))
        else:
            pts=5; details.append(("y","No current dividend; acceptable for reinvestment/growth businesses but not a governance proof."))
        score_sum+=pts; weight_sum+=weight

    # Signal 4: Debt trajectory versus revenue growth
    try:
        bs=t_obj.balance_sheet; inc=t_obj.income_stmt
        if bs is not None and inc is not None and not bs.empty and not inc.empty:
            def _gv(df,names,col=0):
                for n in names:
                    if n in df.index:
                        try:
                            v=float(df.loc[n].iloc[col])
                            if not np.isnan(v): return v
                        except Exception: pass
                return None
            d0=_gv(bs,["Total Debt","Long Term Debt"]); d1=_gv(bs,["Total Debt","Long Term Debt"],1)
            r0=_gv(inc,["Total Revenue","Revenue"]); r1=_gv(inc,["Total Revenue","Revenue"],1)
            if d0 is not None and d1 is not None and r0 and r1 and abs(d1)>EPS and abs(r1)>EPS:
                weight=15; dg=(d0-d1)/abs(d1); rg=(r0-r1)/abs(r1)
                if dg<-0.05:
                    pts=15; details.append(("g",f"Debt declined YoY ({dg*100:+.1f}%)."))
                elif dg<rg:
                    pts=11; details.append(("g",f"Debt growth ({dg*100:+.1f}%) is below revenue growth ({rg*100:+.1f}%)."))
                elif dg<0.1:
                    pts=6; details.append(("y",f"Debt expanded moderately ({dg*100:+.1f}%); verify cash-flow coverage."))
                else:
                    pts=0; details.append(("r",f"Debt expanded faster ({dg*100:+.1f}%) than revenue ({rg*100:+.1f}%)."))
                score_sum+=pts; weight_sum+=weight
    except Exception: pass

    # Signal 5: Cash conversion ratio
    try:
        cf=t_obj.cashflow; inc2=t_obj.income_stmt
        if cf is not None and inc2 is not None and not cf.empty and not inc2.empty:
            def _gcf(df,names):
                for n in names:
                    if n in df.index:
                        try:
                            v=float(df.loc[n].iloc[0])
                            if not np.isnan(v): return v
                        except Exception: pass
                return None
            cfo=_gcf(cf,["Operating Cash Flow","Cash Flow From Continuing Operating Activities","Total Cash From Operating Activities"])
            ni=_gcf(inc2,["Net Income","Net Income Applicable To Common Shares","Net Income From Continuing Operations"])
            if cfo is not None and ni and abs(ni)>EPS:
                weight=10; ccr=cfo/ni
                if ccr>=1.2: pts=10; details.append(("g",f"Excellent cash conversion (CFO/NI {ccr:.2f}x)."))
                elif ccr>=0.8: pts=7; details.append(("g",f"Good cash conversion (CFO/NI {ccr:.2f}x)."))
                elif ccr>=0.5: pts=3; details.append(("y",f"Moderate cash conversion (CFO/NI {ccr:.2f}x)."))
                else: pts=0; details.append(("r",f"Weak cash conversion (CFO/NI {ccr:.2f}x)."))
                score_sum+=pts; weight_sum+=weight
    except Exception: pass

    coverage=round(weight_sum,0)
    if weight_sum==0:
        return {"score":None,"grade":"N/A","details":details,"color":"#3d4466","coverage":0}
    s=round(score_sum/weight_sum*100,1)
    if coverage<50:
        details.insert(0,("y",f"Low CG-Proxy data coverage ({coverage:.0f}%). Use only as a prompt for manual review."))
    return {"score":s,"grade":_score_grade(s),"details":details,"color":_score_color(s),"coverage":coverage}


# ╔══════════════════════════════════════════════════════════════════╗
#  EQ-SCORE™ — EARNINGS QUALITY SCORE (0-100)
# ╚══════════════════════════════════════════════════════════════════╝
def compute_eq_score(t_obj):
    """
    EQ-Score™ (0-100) — earnings-quality proxy.
    Requires at least 3 of 4 components to avoid false precision.

    Component A: CFO / Net Income → cash earnings quality
    Component B: Accruals ratio → non-cash earnings pressure
    Component C: Gross-margin trend → pricing power stability
    Component D: Receivables growth vs revenue growth → collection/channel-stuffing proxy
    """
    score=50.0; details=[]; components=0
    try:
        cf=t_obj.cashflow; inc=t_obj.income_stmt; bs=t_obj.balance_sheet
        if cf is None or inc is None or cf.empty or inc.empty:
            return {"score":None,"grade":"N/A","details":[("y","Insufficient cash-flow/income-statement data for EQ-Score.")],"color":"#3d4466","coverage":0}

        def _gv(df,names,col=0):
            for n in names:
                if n in df.index:
                    try:
                        v=float(df.loc[n].iloc[col])
                        if not np.isnan(v): return v
                    except Exception: pass
            return None

        cfo=_gv(cf,["Operating Cash Flow","Cash Flow From Continuing Operating Activities","Total Cash From Operating Activities"])
        ni=_gv(inc,["Net Income","Net Income Applicable To Common Shares","Net Income From Continuing Operations"])

        # A: CFO / NI ratio
        if cfo is not None and ni and abs(ni)>EPS:
            components+=1; ccr=cfo/ni
            if ccr>=1.3: score+=20; details.append(("g",f"CFO/NI {ccr:.2f}x — profits strongly backed by operating cash flow."))
            elif ccr>=1.0: score+=10; details.append(("g",f"CFO/NI {ccr:.2f}x — profits broadly backed by cash."))
            elif ccr>=0.7: details.append(("y",f"CFO/NI {ccr:.2f}x — acceptable but monitor working capital."))
            else: score-=20; details.append(("r",f"CFO/NI {ccr:.2f}x — weak cash conversion."))

        # B: Accruals Ratio = (NI - CFO) / Avg Total Assets
        if bs is not None and not bs.empty and cfo is not None and ni is not None:
            ta0=_gv(bs,["Total Assets"],0); ta1=_gv(bs,["Total Assets"],1)
            if ta0 and ta1 and (ta0+ta1)>EPS:
                components+=1; accruals=(ni-cfo)/((ta0+ta1)/2)
                if abs(accruals)<0.02: score+=15; details.append(("g",f"Accruals ratio {accruals*100:.2f}% — low non-cash earnings pressure."))
                elif abs(accruals)<0.05: score+=7; details.append(("g",f"Accruals ratio {accruals*100:.2f}% — acceptable range."))
                elif abs(accruals)<0.09: score-=5; details.append(("y",f"Accruals ratio {accruals*100:.2f}% — elevated; monitor."))
                else: score-=15; details.append(("r",f"Accruals ratio {accruals*100:.2f}% — high non-cash earnings pressure."))

        # C: Gross Margin Trend
        r0=_gv(inc,["Total Revenue","Revenue"],0); r1=_gv(inc,["Total Revenue","Revenue"],1)
        g0=_gv(inc,["Gross Profit"],0); g1=_gv(inc,["Gross Profit"],1)
        if r0 and g0 and r0>EPS and r1 and g1 and r1>EPS:
            components+=1; gm0=g0/r0; gm1=g1/r1; delta=gm0-gm1
            if delta>0.02: score+=15; details.append(("g",f"Gross margin expanded by {delta*100:.1f} percentage points."))
            elif delta>-0.01: score+=8; details.append(("g",f"Gross margin broadly stable ({gm0*100:.1f}%)."))
            elif delta>-0.04: score-=5; details.append(("y",f"Gross margin compressed by {abs(delta)*100:.1f} percentage points."))
            else: score-=12; details.append(("r",f"Gross margin compressed sharply by {abs(delta)*100:.1f} percentage points."))

        # D: Receivables growth vs Revenue growth
        if bs is not None and not bs.empty:
            rec0=_gv(bs,["Net Receivables","Receivables"],0); rec1=_gv(bs,["Net Receivables","Receivables"],1)
            if rec0 and rec1 and r0 and r1 and abs(r1)>EPS and abs(rec1)>EPS:
                components+=1; rev_g=(r0-r1)/abs(r1); rec_g=(rec0-rec1)/abs(rec1)
                if rec_g<rev_g-0.05:
                    score+=10; details.append(("g","Receivables grew slower than revenue — collection quality looks better."))
                elif rec_g>rev_g+0.15:
                    score-=12; details.append(("r",f"Receivables grew {rec_g*100:.1f}% vs revenue {rev_g*100:.1f}% — investigate collections/channel sales."))
                else:
                    score+=3; details.append(("y","Receivables broadly in line with revenue."))

    except Exception as e:
        return {"score":None,"grade":"N/A","details":[("y",f"EQ-Score could not be computed reliably: {str(e)[:80]}")],"color":"#3d4466","coverage":0}

    coverage=round(components/4*100,0)
    if components<3:
        details.insert(0,("y",f"Only {components}/4 EQ components available. Score hidden to avoid false precision."))
        return {"score":None,"grade":"N/A","details":details,"color":"#3d4466","coverage":coverage}

    s=round(max(0,min(100,score)),1)
    return {"score":s,"grade":_score_grade(s),"details":details,"color":_score_color(s),"coverage":coverage}


# ╔══════════════════════════════════════════════════════════════════╗
#  PORTFOLIO HEALTH SCORE (0-100)
# ╚══════════════════════════════════════════════════════════════════╝
def compute_portfolio_health(tickers, weights, close, metrics_list):
    breakdown={}
    score=0

    # D1: Diversification — HHI (Herfindahl-Hirschman Index)
    w=np.array([weights.get(t,0) for t in tickers]); w/=w.sum()+EPS
    hhi=float(np.sum(w**2))
    n=len(tickers)
    div_score=max(0,min(35,(1-hhi)*35/(1-1/max(n,2))))
    breakdown["Diversification"]={"score":round(div_score,0),"max":35,
        "detail":f"{n} stocks, HHI={hhi:.2f} (lower=more diversified)"}
    score+=div_score

    # D2: Sector-aware quality — average Alpha Score instead of one-size-fits-all ROE/ROCE
    alpha_vals=[]; model_counts={}
    for m in metrics_list:
        a=compute_sector_alpha_score(m)
        if a.get("score") is not None:
            alpha_vals.append(a["score"])
            model_counts[a["model_name"]]=model_counts.get(a["model_name"],0)+1
    avg_alpha=float(np.mean(alpha_vals)) if alpha_vals else 0
    q_score=min(30, avg_alpha/100*30)
    top_models=", ".join([f"{k}×{v}" for k,v in sorted(model_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]]) or "No usable model data"
    breakdown["Sector-aware Quality"]={"score":round(q_score,0),"max":30,
        "detail":f"Avg Sector Alpha Score: {avg_alpha:.1f}/100 · Models: {top_models}"}
    score+=q_score

    # D3: Risk — historical annualized volatility + max drawdown
    rets=close[tickers].pct_change().dropna()
    port_r=(rets*pd.Series(weights)).sum(axis=1)
    cum=(1+port_r).cumprod(); mdd=float((cum/cum.cummax()-1).min()) if len(cum) else 0
    ann_vol=float(port_r.std()*np.sqrt(ANN)) if len(port_r) else 0
    r_score=max(0,min(25,(1-ann_vol)*12.5+max(0,1+mdd)*12.5))
    breakdown["Risk Management"]={"score":round(r_score,0),"max":25,
        "detail":f"Ann. Vol: {ann_vol*100:.1f}% · Max DD: {mdd*100:.1f}%"}
    score+=r_score

    # D4: Momentum — average distance from 52W high
    fh_vals=[m.get("from_52w_high") for m in metrics_list if m.get("from_52w_high") is not None]
    avg_fh=np.mean(fh_vals) if fh_vals else -20
    m_score=max(0,min(10,(avg_fh+50)/50*10))
    breakdown["Momentum"]={"score":round(m_score,0),"max":10,
        "detail":f"Avg vs 52W High: {avg_fh:+.1f}%"}
    score+=m_score

    total=round(min(100,score),1)
    return {"total":total,"grade":_score_grade(total),"color":_score_color(total),"breakdown":breakdown}


# ╔══════════════════════════════════════════════════════════════════╗
#  AI / LLM FUNCTIONS
# ╚══════════════════════════════════════════════════════════════════╝
# Stronger default for research briefs; fast model retained for low-stakes JSON parsing.
GROQ_RESEARCH_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
ANTHROPIC_RESEARCH_MODEL = "claude-sonnet-4-6"

def call_llm(prompt, system_msg, key, provider, purpose="research", max_tokens=None, temperature=None):
    """Unified LLM caller for Groq or Anthropic.

    purpose="json" keeps the cheap/fast model for natural-language screen parsing.
    purpose="research" uses the stronger research model and a larger output budget.
    """
    if not key: return None
    is_json = purpose == "json"
    max_tokens = max_tokens if max_tokens is not None else (450 if is_json else 1800)
    temperature = temperature if temperature is not None else (0.0 if is_json else 0.2)
    try:
        if provider=="Groq":
            import groq
            c=groq.Groq(api_key=key)
            model = GROQ_FAST_MODEL if is_json else GROQ_RESEARCH_MODEL
            r=c.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system_msg},{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        elif provider=="Anthropic":
            import anthropic
            c=anthropic.Anthropic(api_key=key)
            r=c.messages.create(
                model=ANTHROPIC_RESEARCH_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[{"role":"user","content":prompt}],
            )
            return r.content[0].text
    except Exception as e:
        return f"API Error: {str(e)[:200]}"

def parse_nl_query(query, key, provider):
    """Parse natural language screen query into filter dict using LLM."""
    system="""You are a financial analyst tool. Convert natural language stock screening queries into JSON filter parameters.
Available filters:
- roe_min (number, e.g. 15)
- de_max (number, e.g. 100)
- op_margin_min (number, e.g. 10)
- pe_max (number, e.g. 25)
- pe_min (number)
- pb_max (number, e.g. 3)
- rev_growth_min (number, e.g. 15 for 15%)
- earn_growth_min (number)
- div_yield_min (number, e.g. 2 for 2%)
- beta_max (number, e.g. 0.8)
- roce_min (number)
- analyst_upside_min (number, e.g. 20)
- cg_score_min (number, e.g. 60)
- sector (string, e.g. "Technology", "Healthcare", "Financial Services")

Respond ONLY with a valid JSON object. No explanation, no markdown. Examples:
Query: "quality IT companies with low debt" → {"roe_min": 15, "de_max": 80, "sector": "Technology"}
Query: "cheap value stocks with dividends" → {"pe_max": 18, "div_yield_min": 2}
Query: "high growth companies" → {"rev_growth_min": 20, "earn_growth_min": 15}"""

    resp=call_llm(query, system, key, provider, purpose="json", max_tokens=350, temperature=0.0)
    if not resp: return None
    try:
        match=re.search(r'\{[^}]+\}',resp,re.DOTALL)
        if match: return json.loads(match.group())
    except: pass
    return None


# ╔══════════════════════════════════════════════════════════════════╗
#  DATA FETCHERS
# ╚══════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=86400,show_spinner=False)
def fetch_metrics(ticker):
    base={"ticker":ticker,"name":ticker,"sector":"N/A","error":None,"cg_score":None}
    try:
        t=yf.Ticker(ticker); info=t.info or {}
        if not info or len(info)<5: return {**base,"error":"No data"}
        price=_r(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))

        # Quick CG-Proxy from info only (for screening performance)
        cg_quick=0.0
        risks=[info.get(k) for k in ["auditRisk","boardRisk","compensationRisk","shareHolderRightsRisk"]]
        risks=[r for r in risks if r is not None]
        if risks: cg_quick+=max(0,(10-sum(risks)/len(risks))/10)*50
        ins=info.get("heldPercentInsiders")
        if ins:
            p=float(ins)*100
            cg_quick+=(25 if p>=51 else 18 if p>=30 else 10 if p>=15 else 3)
        dy=info.get("dividendYield"); cg_quick+=(15 if dy and float(dy)>0.01 else 5)
        cg_quick=round(min(cg_quick,80),1)  # Quick proxy capped at 80 (full score needs financial statements)

        m={**base,
            "name":info.get("shortName",ticker),"sector":info.get("sector","N/A"),
            "industry":info.get("industry","N/A"),"market_cap":info.get("marketCap"),
            "price":price,
            "pe_ttm":_r(info.get("trailingPE")),"pe_fwd":_r(info.get("forwardPE")),
            "pb":_r(info.get("priceToBook")),"ev_ebitda":_r(info.get("enterpriseToEbitda")),
            "roe":_p(info.get("returnOnEquity")),"roa":_p(info.get("returnOnAssets")),"op_margin":_p(info.get("operatingMargins")),
            "net_margin":_p(info.get("profitMargins")),"gross_margin":_p(info.get("grossMargins")),
            "rev_growth_ttm":_p(info.get("revenueGrowth")),"earn_growth_ttm":_p(info.get("earningsGrowth")),
            "de_ratio":_r(info.get("debtToEquity")),"current_ratio":_r(info.get("currentRatio")),
            "beta":_r(info.get("beta")),"div_yield":_p(info.get("dividendYield")),
            "year_high":_r(info.get("fiftyTwoWeekHigh")),"year_low":_r(info.get("fiftyTwoWeekLow")),
            "analyst_target":_r(info.get("targetMeanPrice")),"analyst_count":info.get("numberOfAnalystOpinions"),
            "fcf":info.get("freeCashflow"),
            "fcf_yield":None,"analyst_upside":None,"from_52w_high":None,
            "graham_number":None,"graham_mos":None,"roce":None,
            "cg_score":cg_quick,
            "insider_pct":round(float(info.get("heldPercentInsiders",0))*100,1) if info.get("heldPercentInsiders") else None,
            "inst_pct":round(float(info.get("heldPercentInstitutions",0))*100,1) if info.get("heldPercentInstitutions") else None,
        }

        mc,fc=m["market_cap"],m["fcf"]
        if mc and fc and mc>0: m["fcf_yield"]=round(fc/mc*100,2)
        if price and m["analyst_target"] and price>0:
            m["analyst_upside"]=round((m["analyst_target"]-price)/price*100,2)
        if price and m["year_high"] and m["year_high"]>0:
            m["from_52w_high"]=round((price-m["year_high"])/m["year_high"]*100,2)

        eps=info.get("trailingEps"); bvps=info.get("bookValue")
        if eps and bvps and float(eps)>0 and float(bvps)>0:
            gn=(22.5*float(eps)*float(bvps))**0.5
            m["graham_number"]=round(gn,2)
            if price and price>0: m["graham_mos"]=round((gn-price)/gn*100,2)

        try:
            inc_s=t.income_stmt; bs_s=t.balance_sheet
            def _row(df,names):
                for n in names:
                    if n in df.index:
                        try:
                            v=df.loc[n].iloc[0]
                            if v is not None and not (isinstance(v,float) and np.isnan(v)): return float(v)
                        except: pass
                return None
            if inc_s is not None and bs_s is not None and not inc_s.empty and not bs_s.empty:
                ebit=_row(inc_s,["EBIT","Operating Income","Pretax Income"])
                ta=_row(bs_s,["Total Assets","TotalAssets"])
                cl=_row(bs_s,["Current Liabilities","Total Current Liabilities"])
                if ebit and ta and cl:
                    ce=ta-cl
                    if ce>0: m["roce"]=round(ebit/ce*100,2)
        except: pass
        m["data_quality"]=_data_availability_score(m)
        m["data_confidence"]=_data_confidence_label(m["data_quality"])
        alpha=compute_sector_alpha_score(m)
        m["sector_model"]=alpha.get("model_name")
        m["sector_model_key"]=alpha.get("model_key")
        m["alpha_score"]=alpha.get("score")
        m["alpha_grade"]=alpha.get("grade")
        m["alpha_coverage"]=alpha.get("coverage")
    except Exception as e:
        return {**base,"error":str(e)[:80]}
    return m


@st.cache_data(ttl=3600,show_spinner=False)
def fetch_close(tickers,start,end,interval="1d"):
    frames={}
    for ticker in tickers:
        try:
            hist=yf.Ticker(ticker).history(start=start,end=end,interval=interval,auto_adjust=True,actions=False)
            if hist.empty: continue
            s=hist["Close"].copy()
            if s.index.tz: s.index=s.index.tz_localize(None)
            if interval!="1h": s.index=s.index.normalize()
            s.index=pd.to_datetime(s.index); s.name=ticker; frames[ticker]=s
        except: pass
    if not frames: return pd.DataFrame()
    return pd.DataFrame(frames).ffill(limit=5)


@st.cache_data(ttl=900,show_spinner=False)
def fetch_index_snapshot():
    out={}
    for sym,name in INDICES.items():
        try:
            t=yf.Ticker(sym); info=t.info or {}
            p=_r(info.get("regularMarketPrice") or info.get("currentPrice"))
            prev=_r(info.get("regularMarketPreviousClose") or info.get("previousClose"))
            chg=round((p-prev)/prev*100,2) if p and prev and prev>0 else None
            out[name]={"price":p,"chg":chg,"sym":sym}
        except: out[name]={"price":None,"chg":None,"sym":sym}
    return out


@st.cache_data(ttl=3600,show_spinner=False)
def fetch_sector_performance():
    """Returns true 5-trading-session returns using equal-weight sector proxy stocks."""
    end=datetime.now(); start=end-timedelta(days=12)  # buffer for weekends/holidays
    out={}
    for sec,tks in SECTOR_PROXIES.items():
        try:
            cl=fetch_close(tks,start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
            if cl.empty: continue
            rets=[]
            for col in cl.columns:
                s=cl[col].dropna()
                if len(s)>=6:
                    rets.append((s.iloc[-1]/s.iloc[-6]-1)*100)
            out[sec]=round(float(np.mean(rets)),2) if rets else None
        except Exception:
            out[sec]=None
    return out


@st.cache_data(ttl=900,show_spinner=False)
def fetch_close_bulk(tickers,start,end,interval="1d"):
    """Faster close-price fetch for large mover scans. Falls back to empty DataFrame on failure."""
    try:
        data=yf.download(
            tickers=list(tickers), start=start, end=end, interval=interval,
            auto_adjust=True, actions=False, progress=False, threads=True, group_by="ticker"
        )
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns,pd.MultiIndex):
            frames={}
            # yfinance can return columns as (ticker, field) or (field, ticker)
            if "Close" in data.columns.get_level_values(-1):
                for tk in tickers:
                    try:
                        s=data[(tk,"Close")].dropna(); s.name=tk; frames[tk]=s
                    except Exception: pass
            elif "Close" in data.columns.get_level_values(0):
                for tk in tickers:
                    try:
                        s=data[("Close",tk)].dropna(); s.name=tk; frames[tk]=s
                    except Exception: pass
            out=pd.DataFrame(frames)
        else:
            out=pd.DataFrame({tickers[0]:data["Close"]}) if "Close" in data else pd.DataFrame()
        if not out.empty:
            if getattr(out.index,"tz",None): out.index=out.index.tz_localize(None)
            out.index=pd.to_datetime(out.index).normalize()
            out=out.ffill(limit=5)
        return out
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900,show_spinner=False)
def fetch_actual_movers(tickers):
    """Returns actual 1-session top gainers/losers from the selected universe."""
    end=datetime.now(); start=end-timedelta(days=10)
    cl=fetch_close_bulk(tuple(tickers),start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
    if cl.empty:
        cl=fetch_close(tickers,start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
    if cl.empty:
        return [], [], {"usable":0,"total":len(tickers)}
    day_rets={}
    for t in cl.columns:
        s=cl[t].dropna()
        if len(s)>=2 and s.iloc[-2]!=0:
            day_rets[t]=round((s.iloc[-1]/s.iloc[-2]-1)*100,2)
    sorted_r=sorted(day_rets.items(),key=lambda x:x[1],reverse=True)
    return sorted_r[:5], sorted_r[-5:], {"usable":len(day_rets),"total":len(tickers)}


def run_screen(lens_name, custom_filters=None, universe=None):
    # Use passed universe, or fall back to Nifty 200
    scan_tickers = universe if universe else NIFTY_200

    cache_key=f"{lens_name}|{len(scan_tickers)}|{hash(tuple(scan_tickers[:10]))}"
    cache=st.session_state.lens_cache.get(cache_key,{})
    if (cache.get("df") is not None and custom_filters is None and
        (datetime.now()-cache.get("ts",datetime.min)).seconds<21600):
        return cache["df"]

    lens=LENSES.get(lens_name,{})
    filters=custom_filters if custom_filters else None
    n_tickers=len(scan_tickers)
    results=[]; failed=0; skipped=0; prog=st.progress(0.0,text=f"Scanning {n_tickers} stocks…")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(fetch_metrics,t):t for t in scan_tickers}
        done=0
        for fut in as_completed(futs):
            done+=1; prog.progress(done/n_tickers,text=f"Scanning {done}/{n_tickers} stocks…")
            m=fut.result()
            if m.get("error"):
                failed+=1; continue
            if not m.get("name") or m["name"]==m["ticker"]:
                skipped+=1; continue

            if not passes_sector_lens(m, lens_name, custom_filters=filters):
                continue

            alpha=compute_sector_alpha_score(m, lens_name)
            if alpha.get("score") is None:
                skipped+=1; continue
            m["_score"]=round(alpha["score"]/10,1)     # table shows 0-10
            m["sector_score"]=alpha["score"]           # full 0-100 score
            m["sector_score_grade"]=alpha["grade"]
            m["sector_score_coverage"]=alpha["coverage"]
            m["sector_model"]=alpha["model_name"]
            m["sector_model_key"]=alpha["model_key"]
            m["sector_score_components"]=alpha["components"]
            results.append(m)

    prog.empty()
    st.caption(f"Scanned {n_tickers} stocks · sector-aware matches {len(results)} · data failures {failed} · skipped {skipped}. yfinance demo data can be incomplete.")
    if not results: return pd.DataFrame()
    results.sort(key=lambda x:x.get("sector_score",0),reverse=True)
    df=pd.DataFrame(results[:25])
    if custom_filters is None:
        st.session_state.lens_cache[cache_key]={"df":df,"ts":datetime.now()}
    return df


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: MARKET PULSE
# ╚══════════════════════════════════════════════════════════════════╝
def page_market_pulse():
    st.markdown("## 🏠 Market Pulse")
    st.caption(f"NSE market snapshot · {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}")

    # ── Index snapshot
    with st.spinner("Fetching index data…"):
        idx=fetch_index_snapshot()

    c1,c2,c3,c4,c5,c6=st.columns(6)
    for col,(name,d) in zip([c1,c2,c3,c4,c5,c6],idx.items()):
        p=d["price"]; chg=d["chg"]
        with col:
            col.metric(
                name,
                f"{p:,.0f}" if p else "—",
                delta=f"{chg:+.2f}%" if chg else None,
            )

    st.divider()

    # ── Sector heatmap
    left,right=st.columns([1,1])
    with left:
        st.markdown("### Sector Performance (5D)")
        with st.spinner("Loading sector data…"):
            sector_perf=fetch_sector_performance()

        if sector_perf:
            sectors=list(sector_perf.keys()); vals=list(sector_perf.values())
            colors=["#22c55e" if v and v>=0 else "#ef4444" for v in vals]
            fig=go.Figure(go.Bar(
                x=vals, y=sectors, orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}%" if v else "—" for v in vals],
                textposition="auto",
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            _dark(fig,h=320,title="Equal-Weight Sector Returns (5-Day)")
            fig.update_layout(xaxis_title="Return %",showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

    # ── Actual top movers from selected universe
    with right:
        st.markdown("### Actual Top Movers Today")
        mover_choice=st.selectbox(
            "Mover universe", list(UNIVERSE_PRESETS.keys()), index=1,
            key="pulse_mover_universe", label_visibility="collapsed"
        )
        mover_list=_get_universe(mover_choice)
        st.caption(f"Actual 1-session movers from {mover_choice}; not a hand-picked sample.")
        with st.spinner(f"Loading movers for {len(mover_list)} stocks…"):
            top5,bot5,mover_meta=fetch_actual_movers(mover_list)

        if top5 or bot5:
            st.caption(f"Usable price data: {mover_meta['usable']}/{mover_meta['total']} stocks")
            g_col,r_col=st.columns(2)
            with g_col:
                st.markdown('<span class="lbl">Top Gainers</span>',unsafe_allow_html=True)
                for tk,ret in top5:
                    st.markdown(f'<div class="flag-g">🟢 {tk.replace(".NS","")} &nbsp;<span class="mono">{ret:+.2f}%</span></div>',
                                unsafe_allow_html=True)
            with r_col:
                st.markdown('<span class="lbl">Top Losers</span>',unsafe_allow_html=True)
                for tk,ret in bot5:
                    st.markdown(f'<div class="flag-r">🔴 {tk.replace(".NS","")} &nbsp;<span class="mono">{ret:+.2f}%</span></div>',
                                unsafe_allow_html=True)
        else:
            st.warning("Could not calculate movers from current yfinance data.")

    st.divider()

    # ── Nifty 50 mini chart
    st.markdown("### Nifty 50 — 30-Day Chart")
    with st.spinner("Loading Nifty chart…"):
        end=datetime.now(); start=end-timedelta(days=45)
        nifty_cl=fetch_close(["^NSEI"],start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))

    if not nifty_cl.empty and "^NSEI" in nifty_cl.columns:
        s=nifty_cl["^NSEI"].dropna()
        if len(s)>=2:
            chg=(s.iloc[-1]/s.iloc[0]-1)*100; lc="#22c55e" if chg>=0 else "#ef4444"
            rsi_nifty=compute_rsi(s)
            macd_n,sig_n,_=compute_macd(s)
            bb_up,bb_mid,bb_lo=compute_bb(s)

            fig=go.Figure()
            fig.add_trace(go.Scatter(x=s.index,y=s.round(0),mode="lines",name="Nifty 50",
                line=dict(color=lc,width=2),fill="tozeroy",fillcolor=f"rgba({','.join(str(int(lc[i:i+2],16)) for i in (1,3,5))},0.07)",
                hovertemplate="%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=s.index,y=bb_up.round(0),mode="lines",name="BB Upper",
                line=dict(color="#374151",width=1,dash="dot"),hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s.index,y=bb_lo.round(0),mode="lines",name="BB Lower",
                line=dict(color="#374151",width=1,dash="dot"),
                fill="tonexty",fillcolor="rgba(55,65,81,0.05)",hoverinfo="skip"))
            _dark(fig,h=240)
            fig.update_layout(showlegend=False,hovermode="x unified",
                annotations=[dict(text=f"Nifty {chg:+.1f}% · RSI {rsi_nifty.iloc[-1]:.0f}",
                    x=0.02,y=0.92,xref="paper",yref="paper",
                    font=dict(size=13,color=lc),showarrow=False)])
            st.plotly_chart(fig,use_container_width=True)

    # ── Market Intelligence cards
    st.divider()
    st.markdown("### Market Intelligence")
    mi1,mi2,mi3=st.columns(3)
    with mi1:
        vix_d=idx.get("India VIX",{}); vix_p=vix_d.get("price")
        vix_chg=vix_d.get("chg")
        if vix_p:
            if vix_p<15: mood="😴 Complacent"; mc="#22c55e"; tip="Low fear. Markets may be complacent. Good time for quality stocks, be cautious on aggressive entries."
            elif vix_p<20: mood="😐 Calm"; mc="#6b7280"; tip="Normal volatility regime. Continue systematic investing."
            elif vix_p<25: mood="😟 Cautious"; mc="#f59e0b"; tip="Elevated volatility. Be selective. Hold cash, avoid leveraged positions."
            else: mood="😱 Fear Zone"; mc="#ef4444"; tip="High fear. Historically a good time to buy quality names for long-term investors."
            st.markdown(f'<div class="gauge-wrap"><div class="gauge-score" style="color:{mc}">{vix_p:.1f}</div>'
                f'<div class="gauge-grade" style="color:{mc}">{mood}</div>'
                f'<div class="gauge-label">India VIX (Volatility Index)</div>'
                f'<div style="font-size:12px;color:#6b7280;margin-top:8px;line-height:1.5">{tip}</div></div>',
                unsafe_allow_html=True)
    with mi2:
        nifty_d=idx.get("Nifty 50",{}); bn_d=idx.get("Bank Nifty",{})
        bn_chg=bn_d.get("chg"); n_chg=nifty_d.get("chg")
        rel_signal="Positive" if (bn_chg and n_chg and bn_chg>n_chg) else "Neutral" if (bn_chg and n_chg and abs(bn_chg-n_chg)<0.3) else "Caution"
        fc="#22c55e" if rel_signal=="Positive" else "#f59e0b" if rel_signal=="Neutral" else "#ef4444"
        st.markdown(f'<div class="gauge-wrap"><div class="gauge-score" style="color:{fc}">{rel_signal}</div>'
            f'<div class="gauge-grade" style="color:#6b7280">Relative Strength</div>'
            f'<div class="gauge-label">Bank Nifty vs Nifty</div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:8px;line-height:1.5">'
            f'Bank Nifty: {bn_chg:+.2f}% · Nifty: {n_chg:+.2f}%<br>'
            f'Bank outperformance is a relative-strength signal only, not proof of institutional buying.</div></div>',
            unsafe_allow_html=True)
    with mi3:
        it_d=idx.get("IT Index",{}); it_chg=it_d.get("chg")
        n_ch=nifty_d.get("chg",0) or 0; it_ch=it_chg or 0
        beat=it_ch-n_ch
        it_c="#22c55e" if beat>0.5 else "#f59e0b" if beat>-0.5 else "#ef4444"
        st.markdown(f'<div class="gauge-wrap"><div class="gauge-score" style="color:{it_c}">{it_ch:+.1f}%</div>'
            f'<div class="gauge-grade" style="color:{it_c}">{"Outperforming" if beat>0.3 else "In Line" if beat>-0.3 else "Lagging"}</div>'
            f'<div class="gauge-label">IT Index vs Nifty</div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:8px;line-height:1.5">'
            f'IT sector is {beat:+.1f}pp vs benchmark today.<br>Watch for USD/INR correlation signals.</div></div>',
            unsafe_allow_html=True)


def _get_universe(choice: str) -> list:
    """Resolve a universe preset name to a list of tickers. MVP scan universe is capped at Nifty 500."""
    if "Nifty 50" in choice: return NIFTY_200[:50]
    if "Nifty 500" in choice: return _universe(NIFTY_500_EXT)
    return NIFTY_200


# ── Navigation helpers ──────────────────────────────────────────────────────
# Streamlit widgets keep their own state. When a button jumps from one page to
# another, we must update BOTH the internal page state and the top radio widget
# state, otherwise the radio can immediately switch the app back to the old page.
NAV_LABELS = [
    "🏠 Market Pulse", "🤖 AI Screener", "🔬 Deep Dive",
    "💼 Portfolio Lab", "🚨 Risk Radar", "📘 Metrics Guide",
]
PAGE_MAP = {
    "🏠 Market Pulse":"pulse", "🤖 AI Screener":"screen", "🔬 Deep Dive":"analyze",
    "💼 Portfolio Lab":"portfolio", "🚨 Risk Radar":"radar", "📘 Metrics Guide":"metrics",
}
PAGE_TO_NAV = {v:k for k,v in PAGE_MAP.items()}

def _normalise_ticker(ticker: str) -> str:
    t = str(ticker or "TCS.NS").strip().upper()
    if not t:
        return "TCS.NS"
    if t.endswith((".NS", ".BO")):
        return t
    # Default to NSE for Indian tickers typed without an exchange suffix.
    return f"{t}.NS"

def _go_to(page: str, ticker: str | None = None):
    """Request a page jump on the next rerun.

    Important Streamlit detail: the top navigation radio is already created by
    the time row-level buttons are clicked. Updating the radio widget key
    (main_nav) after creation is unreliable and can make the app bounce back to
    the previous page. So row buttons set pending non-widget state only. The
    pending jump is applied BEFORE the radio is rendered on the next run.
    """
    st.session_state._pending_page = page
    if ticker is not None:
        st.session_state._pending_ticker = _normalise_ticker(ticker)
    st.rerun()

def _apply_pending_navigation():
    """Apply pending programmatic navigation before navigation widgets render."""
    page = st.session_state.pop("_pending_page", None)
    ticker = st.session_state.pop("_pending_ticker", None)
    if ticker is not None:
        tk = _normalise_ticker(ticker)
        st.session_state.analyze_ticker = tk
        # This is safe here because the Deep Dive text input has not yet been
        # instantiated in this run.
        st.session_state.dd_ticker_in = tk
    if page is not None:
        st.session_state.page = page
        st.session_state.main_nav = PAGE_TO_NAV.get(page, NAV_LABELS[0])


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: AI SCREENER
# ╚══════════════════════════════════════════════════════════════════╝
def page_ai_screener():
    st.markdown("## 🤖 AI Stock Screener")
    st.caption("Discover stocks in plain English, pick a lens, or search a company by name. MVP scans are capped at Nifty 500; yfinance search can still help find other tickers.")

    tab1,tab2,tab3=st.tabs(["🔎 Find Any Stock","💬 Natural Language","🔭 Investment Lenses"])

    # ── TAB 1: FIND ANY STOCK ────────────────────────────────────────────
    with tab1:
        st.markdown("#### Search Any Listed Company")
        st.markdown(
            '<div class="flag-g">✨ <strong>Company Search</strong> — searches yfinance for NSE/BSE tickers. '
            'The screener itself is capped at Nifty 500 for this MVP. '
            'Type a name, get the ticker, then verify data availability.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        search_query = st.text_input(
            "", placeholder="Type company name — e.g. Kaynes Technology, Nykaa, Delhivery, PB Fintech...",
            label_visibility="collapsed", key="stock_search_input",
        ).strip()

        if search_query:
            with st.spinner(f"Searching NSE for '{search_query}'…"):
                results = search_nse_stocks(search_query)

            if results:
                st.markdown(f'<span class="lbl">Found {len(results)} match(es)</span>', unsafe_allow_html=True)
                for r in results:
                    col1, col2, col3 = st.columns([4, 2, 2])
                    with col1:
                        tags = f'<span class="pill pill-b">{r["ticker"]}</span> '
                        tags += f'<span style="font-size:14px;font-weight:600;color:#e2e6ff">{r["name"]}</span>'
                        if r.get("sector"): tags += f' <span class="pill pill-p">{r["sector"][:20]}</span>'
                        tags += f' <span class="pill {"pill-g" if r["exchange"]=="NSE" else "pill-y"}">{r["exchange"]}</span>'
                        st.markdown(tags, unsafe_allow_html=True)
                    with col2:
                        if st.button(f"🔬 Deep Dive", key=f"dd_{r['ticker']}", use_container_width=True, type="primary"):
                            _go_to("analyze", r["ticker"])
                    with col3:
                        if st.button(f"➕ Portfolio", key=f"pf_{r['ticker']}", use_container_width=True):
                            if r["ticker"] not in st.session_state.portfolio_tickers:
                                st.session_state.portfolio_tickers.append(r["ticker"])
                                st.session_state.portfolio_data = None
                            st.success(f"Added {r['ticker']}")
                    st.markdown("---")

                # Load fundamentals of top result inline
                top = results[0]["ticker"]
                with st.spinner(f"Loading fundamentals for {top}…"):
                    m = fetch_metrics(top)
                if not m.get("error"):
                    st.markdown(f"#### Quick Snapshot: {m.get('name', top)}")
                    q1,q2,q3,q4,q5,q6 = st.columns(6)
                    q1.metric("Price", f"₹{m['price']:,.2f}" if m.get("price") else "—")
                    q2.metric("Mkt Cap", _cap(m.get("market_cap")))
                    q3.metric("P/E TTM", f"{m['pe_ttm']:.1f}x" if m.get("pe_ttm") else "—")
                    q4.metric("ROE", f"{m['roe']:.1f}%" if m.get("roe") else "—")
                    q5.metric("Op Margin", f"{m['op_margin']:.1f}%" if m.get("op_margin") else "—")
                    q6.metric("Rev Growth", f"{m['rev_growth_ttm']:+.1f}%" if m.get("rev_growth_ttm") else "—")
            else:
                st.warning(
                    f"No NSE results found for **'{search_query}'**. Tips:\n"
                    "- Use the official company name (not brand name)\n"
                    "- Try partial name: 'Kaynes' instead of 'Kaynes Technology'\n"
                    "- If you know the ticker, type it directly in Deep Dive (e.g. `KAYNES.NS`)"
                )

        st.divider()
        # Manual ticker entry fallback
        st.markdown("#### Or enter the NSE ticker directly")
        st.caption("Use this if you already know the ticker symbol.")
        mc1, mc2 = st.columns([4, 2])
        with mc1:
            manual_tk = st.text_input("", placeholder="KAYNES.NS · NYKAA.NS · DELHIVERY.NS",
                label_visibility="collapsed", key="manual_ticker_input").strip().upper()
        with mc2:
            if manual_tk and st.button("🔬 Analyze →", type="primary", use_container_width=True):
                ticker_clean = manual_tk if manual_tk.endswith((".NS", ".BO")) else f"{manual_tk}.NS"
                _go_to("analyze", ticker_clean)

    # ── TAB 2: NATURAL LANGUAGE ──────────────────────────────────────────
    with tab2:
        st.markdown("#### Ask in plain English")
        examples = [
            "Quality IT companies with low debt and high ROE",
            "Defensive dividend payers with beta under 0.8",
            "High growth companies with strong operating margins",
            "Value stocks with Graham margin of safety",
            "Capital efficient PSU companies with high ROCE",
        ]
        st.caption("Click an example to fill the query box, then press Screen Now.")
        if "nl_input" not in st.session_state:
            st.session_state["nl_input"] = ""
        ex_cols = st.columns(3)
        for i, ex in enumerate(examples[:3]):
            with ex_cols[i]:
                if st.button(f"📝 {ex[:35]}…", key=f"nl_example_{i}", use_container_width=True):
                    # The text_area below is keyed as nl_input. Updating the same
                    # session-state key before the widget is rendered makes the
                    # example buttons visibly populate the input box on click.
                    st.session_state["nl_input"] = ex
                    st.rerun()

        query = st.text_area("", placeholder="E.g. 'Find quality pharma companies with debt-to-equity below 1, ROE above 15%'",
            height=80, key="nl_input", label_visibility="collapsed").strip()

        # Universe selector for NL search
        nl_universe_choice = st.selectbox("Scan universe",
            list(UNIVERSE_PRESETS.keys()), index=1, key="nl_universe")

        key = st.session_state.api_key; provider = st.session_state.api_provider
        if query:
            if not key:
                st.warning("🔑 Add your API key (top bar) to enable natural language screening.")
            else:
                if st.button("🔍 Screen Now", type="primary", key="nl_screen_now"):
                    with st.spinner("Parsing your query with AI…"):
                        parsed = parse_nl_query(query, key, provider)
                    if parsed:
                        st.markdown(f'<div class="flag-g">✅ AI parsed: <code>{json.dumps(parsed)}</code></div>',
                            unsafe_allow_html=True)
                        u = _get_universe(nl_universe_choice)
                        with st.spinner(f"Scanning {len(u)} stocks…"):
                            df = run_screen("_custom", custom_filters=parsed, universe=u)
                        if not df.empty:
                            # Persist results. Otherwise on the next click
                            # Streamlit reruns, the Screen Now button is no
                            # longer active, results disappear, and row action
                            # buttons cannot fire.
                            st.session_state.nl_results_df = df
                            st.session_state.nl_results_title = f"Results for: {query[:60]}"
                            st.session_state.nl_results_query = query
                            st.session_state.nl_results_universe = nl_universe_choice
                        else:
                            st.session_state.nl_results_df = None
                            st.session_state.nl_results_title = None
                            st.warning("No stocks matched. Try relaxing your criteria.")
                    else:
                        st.error("Couldn't parse. Try rephrasing or use the Lenses tab.")

        # Always re-render the latest Natural Language results while this tab is
        # being computed. This makes the stock selectbox and Deep Dive button
        # work across Streamlit reruns.
        if st.session_state.get("nl_results_df") is not None:
            df_saved = st.session_state.nl_results_df
            if isinstance(df_saved, pd.DataFrame) and not df_saved.empty:
                st.divider()
                meta = st.session_state.get("nl_results_universe") or "saved universe"
                st.caption(f"Showing saved Natural Language screener results · {meta}")
                _render_screen_results(
                    df_saved,
                    st.session_state.get("nl_results_title") or "Saved Natural Language results",
                    lens=None,
                    key_prefix="nl_results_saved"
                )
                if st.button("Clear Natural Language results", key="nl_clear_results"):
                    st.session_state.nl_results_df = None
                    st.session_state.nl_results_title = None
                    st.session_state.nl_results_query = None
                    st.session_state.nl_results_universe = None
                    st.rerun()

    # ── TAB 3: INVESTMENT LENSES ─────────────────────────────────────────
    with tab3:
        # Universe selector
        lens_universe_choice = st.selectbox(
            "📡 Scan Universe",
            list(UNIVERSE_PRESETS.keys()), index=1,
            key="lens_universe",
            help="MVP scan universe is capped at Nifty 500 for speed and data-quality control."
        )
        cols = st.columns(4)
        lens_items = list(LENSES.items())
        for i, (ln, lens) in enumerate(lens_items):
            active = st.session_state.active_lens == ln
            border = f"2px solid {lens['color']}" if active else "1.5px solid #1a1e35"
            bg = "#0d1428" if active else "#0c0e1c"
            with cols[i % 4]:
                st.markdown(
                    f'<div style="background:{bg};border:{border};border-radius:12px;'
                    f'padding:14px;margin-bottom:6px;min-height:90px">'
                    f'<div style="font-size:20px">{ln.split()[0]}</div>'
                    f'<div style="font-size:13px;font-weight:700;color:#e2e6ff;margin-top:2px">{lens["title"]}</div>'
                    f'<div style="font-size:11px;color:#4b5675;margin-top:3px;line-height:1.4">{lens["desc"]}</div>'
                    f'</div>', unsafe_allow_html=True)
                if st.button("Screen →", key=f"ln_{i}", use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.active_lens = ln
                    st.session_state.lens_universe_sel = lens_universe_choice
                    st.rerun()

        if not st.session_state.active_lens: return

        ln = st.session_state.active_lens; lens = LENSES[ln]
        u_choice = st.session_state.get("lens_universe_sel", lens_universe_choice)
        st.divider()
        st.markdown(f"### {ln} — {lens['title']}")
        st.markdown(
            f'<div style="background:#0c0e1c;border-left:3px solid {lens["color"]};'
            f'border-radius:0 8px 8px 0;padding:11px 16px;margin-bottom:16px;font-size:13px;color:#9ca3af">'
            f'{lens["detail"]}</div>', unsafe_allow_html=True)

        u = _get_universe(u_choice)
        with st.spinner(f"Scanning {len(u)} stocks… (cached for 6h on Nifty 200)"):
            df = run_screen(ln, universe=u if "Nifty 200" not in u_choice else None)
        if df.empty:
            st.warning("No stocks matched. Try a larger universe or different lens.")
            return
        _render_screen_results(df, f"{len(df)} stocks found · {u_choice}", lens=lens, key_prefix=f"lens_{re.sub(r'[^A-Za-z0-9]+', '_', ln)}_{re.sub(r'[^A-Za-z0-9]+', '_', u_choice)}")


def _render_screen_results(df,title,lens,key_prefix="screen"):
    safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key_prefix))[:90] or "screen"
    st.markdown(f'<span class="lbl">{title}</span>',unsafe_allow_html=True)
    lens_cols=lens["show_cols"] if lens else ["roe","op_margin","de_ratio","pe_ttm"]
    lens_lbls=lens["labels"] if lens else ["ROE %","Op Margin %","D/E","P/E"]

    rows=[]
    for _,row in df.iterrows():
        r={"Score":row.get("_score",0),
           "Alpha 100":row.get("sector_score"),
           "Model":(row.get("sector_model") or "")[:22],
           "Ticker":row["ticker"].replace(".NS","").replace(".BO",""),
           "Company":(row.get("name","") )[:22],
           "Sector":(row.get("sector","N/A") or "N/A")[:14],
           "Mkt Cap":_cap(row.get("market_cap")),
           "Data":f"{row['data_quality']:.0f}%" if row.get("data_quality") is not None else "—",
           "CG-Proxy":f"{row['cg_score']:.0f}" if row.get("cg_score") else "—"}
        for col,lbl in zip(lens_cols,lens_lbls):
            r[lbl]=row.get(col)
        rows.append(r)
    disp=pd.DataFrame(rows)

    def _fv(v,lbl=""):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        if isinstance(v,(int,float)):
            if any(s in lbl for s in ["Growth","MOS","Upside","Yield","ROCE","Margin","ROE","52W"]): return f"{v:+.1f}%"
            return f"{v:.1f}"
        return str(v)

    fmt={**{"Score":"{:.1f}"}}, {**{lbl:lambda v,l=lbl:_fv(v,l) for lbl in lens_lbls}}
    fmt_all={"Score":"{:.1f}", "Alpha 100":"{:.0f}"}
    for lbl in lens_lbls: fmt_all[lbl]=lambda v,l=lbl: _fv(v,l)

    def s_bg(val):
        if not isinstance(val,(int,float)): return ""
        if val>=7.5: return "background:#031a0b;color:#34d399;font-weight:700"
        if val>=5.0: return "background:#1a1208;color:#fbbf24;font-weight:600"
        return "background:#1a0505;color:#f87171"
    def pn(val):
        if not isinstance(val,(int,float)): return ""
        return "color:#22c55e;font-weight:500" if val>=0 else "color:#ef4444;font-weight:500"

    pn_cols=[l for l in lens_lbls if any(x in l for x in ["Growth","MOS","Upside","52W","Yield"])]
    styled=disp.style.map(s_bg,subset=["Score"])
    if "Alpha 100" in disp.columns: styled=styled.map(lambda v: s_bg(v/10 if isinstance(v,(int,float)) else v), subset=["Alpha 100"])
    if pn_cols: styled=styled.map(pn,subset=pn_cols)
    styled=styled.format(fmt_all,na_rep="—")
    st.dataframe(styled,use_container_width=True,height=min(700,60+len(disp)*38))

    # Action row
    st.divider()
    ticker_opts=[f"{r['Ticker']} — {r['Company']}" for r in rows]
    sel_c,btn_c=st.columns([4,2])
    with sel_c:
        chosen=st.selectbox("Analyze a stock",ticker_opts,label_visibility="collapsed",key=f"{safe_key}_sel")
    with btn_c:
        if st.button("🔬 Deep Dive →",type="primary",use_container_width=True,key=f"{safe_key}_deep_btn"):
            chosen_tk=df.iloc[ticker_opts.index(chosen)]["ticker"]
            _go_to("analyze", chosen_tk)

    add_multi=st.multiselect("Add to Portfolio Lab",options=ticker_opts,
        placeholder="Select multiple stocks…",label_visibility="collapsed",key=f"{safe_key}_add")
    if add_multi and st.button("➕ Add to Portfolio Lab", key=f"{safe_key}_add_btn"):
        for opt in add_multi:
            t=df.iloc[ticker_opts.index(opt)]["ticker"]
            if t not in st.session_state.portfolio_tickers:
                st.session_state.portfolio_tickers.append(t)
        st.session_state.portfolio_data=None
        _go_to("portfolio")


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: DEEP DIVE
# ╚══════════════════════════════════════════════════════════════════╝
def page_deep_dive():
    st.markdown("## 🔬 Deep Dive")

    c1,c2=st.columns([3,6])
    with c1:
        # Initialise only once; programmatic Deep Dive buttons update this key
        # before rerunning so the selected stock is shown correctly.
        if "dd_ticker_in" not in st.session_state:
            st.session_state.dd_ticker_in = st.session_state.analyze_ticker or "TCS.NS"
        ti=st.text_input("",
            placeholder="TCS.NS · RELIANCE.NS",label_visibility="collapsed",key="dd_ticker_in").strip().upper()
        if ti:
            ticker_typed = _normalise_ticker(ti)
            if ticker_typed != st.session_state.analyze_ticker:
                st.session_state.analyze_ticker = ticker_typed
    with c2:
        tf=st.radio("",["3M","6M","1Y","3Y","5Y"],horizontal=True,label_visibility="collapsed",key="dd_tf")

    ticker=st.session_state.analyze_ticker or "TCS.NS"
    days_map={"3M":90,"6M":180,"1Y":365,"3Y":1095,"5Y":1825}
    d_end=datetime.now(); d_start=d_end-timedelta(days=days_map[tf])

    with st.spinner(f"Loading {ticker}…"):
        m=fetch_metrics(ticker)
        close=fetch_close([ticker],d_start.strftime("%Y-%m-%d"),d_end.strftime("%Y-%m-%d"))

    if m.get("error"):
        st.error(f"Could not load **{ticker}**: {m['error']}")
        return

    t_obj=yf.Ticker(ticker)
    prices=close[ticker].dropna() if (not close.empty and ticker in close.columns) else pd.Series()

    name=m.get("name",ticker); price=m.get("price"); sector=m.get("sector","N/A")
    fh=m.get("from_52w_high")

    # ── Header ────────────────────────────────────────────────────────────
    h1,h2=st.columns([5,4])
    with h1:
        st.markdown(f"### {name}")
        tags=f'<span class="pill pill-b">{ticker.replace(".NS","")}</span> '
        tags+=f'<span class="pill pill-p">{sector}</span>'
        if fh: tags+=f' <span class="pill {"pill-g" if fh>-10 else "pill-r"}">{fh:+.1f}% vs 52W High</span>'
        st.markdown(tags,unsafe_allow_html=True)
        ps=f"₹{price:,.2f}" if price else "—"
        st.markdown(f'<div style="font-size:36px;font-weight:900;color:#e2e6ff;margin:8px 0;font-family:JetBrains Mono,monospace">{ps}</div>'
            f'<span style="font-size:12px;color:#4b5675">Market Cap: {_cap(m.get("market_cap"))}</span>'
            f'<span style="font-size:12px;color:#4b5675;margin-left:16px">Insider: {m.get("insider_pct","—"):.1f}% · Inst: {m.get("inst_pct","—"):.1f}%</span>'
            if m.get("insider_pct") and m.get("inst_pct") else
            f'<div style="font-size:36px;font-weight:900;color:#e2e6ff;margin:8px 0;font-family:JetBrains Mono,monospace">{ps}</div>'
            f'<span style="font-size:12px;color:#4b5675">Market Cap: {_cap(m.get("market_cap"))}</span>',
            unsafe_allow_html=True)
    with h2:
        a1,a2=st.columns(2)
        a1.metric("52W High",f"₹{m['year_high']:,.0f}" if m.get("year_high") else "—")
        a2.metric("52W Low",f"₹{m['year_low']:,.0f}" if m.get("year_low") else "—")
        a1.metric("Analyst Target",f"₹{m['analyst_target']:,.0f}" if m.get("analyst_target") else "—",
            delta=f"{m['analyst_upside']:+.1f}% upside" if m.get("analyst_upside") else None)
        a2.metric("Data",f"{m.get('data_quality',0):.0f}%" if m.get("data_quality") is not None else "—",
            help="Share of key fields available from yfinance for this app's calculations.")
        alpha=compute_sector_alpha_score(m)
        a1.metric("Sector Alpha", f"{alpha['score']:.0f}/100" if alpha.get("score") is not None else "—",
            delta=f"Grade {alpha.get('grade','N/A')} · {alpha.get('model_name','')}" if alpha.get("score") is not None else None,
            help="Sector-aware score. Uses different metric weights for banks/NBFCs, IT, pharma, industrials, consumer and cyclicals.")

    # ── Price chart + Technical overlays ─────────────────────────────────
    tech={}
    if len(prices)>=50:
        tech=get_technical_signals(prices)
        sma50=prices.rolling(50).mean()
        sma200=prices.rolling(200).mean() if len(prices)>=200 else None
        bb_up,_,bb_lo=compute_bb(prices)
        chg=(prices.iloc[-1]/prices.iloc[0]-1)*100; lc="#22c55e" if chg>=0 else "#ef4444"

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=prices.index,y=prices.round(2),mode="lines",name=name,
            line=dict(color=lc,width=2),hovertemplate="₹%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sma50.index,y=sma50.round(2),mode="lines",name="50 SMA",
            line=dict(color="#f59e0b",width=1),hovertemplate="%{y:,.2f}<extra></extra>"))
        if sma200 is not None:
            fig.add_trace(go.Scatter(x=sma200.index,y=sma200.round(2),mode="lines",name="200 SMA",
                line=dict(color="#8b5cf6",width=1),hovertemplate="%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=bb_up.index,y=bb_up.round(2),mode="lines",name="BB Upper",
            line=dict(color="#374151",width=1,dash="dot"),hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bb_lo.index,y=bb_lo.round(2),mode="lines",name="BB Lower",
            line=dict(color="#374151",width=1,dash="dot"),
            fill="tonexty",fillcolor="rgba(55,65,81,0.06)",hoverinfo="skip"))
        _dark(fig,h=290)
        fig.update_layout(showlegend=True,hovermode="x unified",
            legend=dict(orientation="h",y=1.08,font=dict(size=10)),
            annotations=[dict(text=f"{chg:+.1f}% over period",
                x=0.02,y=0.92,xref="paper",yref="paper",
                font=dict(size=13,color=lc),showarrow=False)])
        st.plotly_chart(fig,use_container_width=True)

        # Technical signal pills
        tc1,tc2,tc3,tc4=st.columns(4)
        tc1.markdown(f'<span class="lbl">RSI (14)</span><div style="font-size:22px;font-weight:700;font-family:JetBrains Mono,monospace;color:{tech.get("rsi_col","#6b7280")}">'
            f'{tech.get("rsi","—")}</div><div style="font-size:12px;color:{tech.get("rsi_col","#6b7280")}">{tech.get("rsi_sig","—")}</div>',
            unsafe_allow_html=True)
        tc2.markdown(f'<span class="lbl">MACD Signal</span><div style="font-size:18px;font-weight:700;color:{tech.get("macd_col","#6b7280")}">'
            f'{tech.get("macd_sig","—")}</div><div style="font-size:11px;color:#4b5675">Hist: {tech.get("macd_hist","—")}</div>',
            unsafe_allow_html=True)
        tc3.markdown(f'<span class="lbl">SMA Trend</span><div style="font-size:15px;font-weight:600;color:{tech.get("sma_col","#6b7280")}">'
            f'{tech.get("sma_sig","—")}</div>'
            f'<div style="font-size:11px;color:#4b5675">50D: ₹{tech.get("sma50",0):,.0f}</div>',
            unsafe_allow_html=True)
        tc4.markdown(f'<span class="lbl">Bollinger %B</span><div style="font-size:22px;font-weight:700;font-family:JetBrains Mono,monospace;color:{tech.get("bb_col","#6b7280")}">'
            f'{tech.get("bb_pct","—")}</div><div style="font-size:12px;color:{tech.get("bb_col","#6b7280")}">{tech.get("bb_sig","—")}</div>',
            unsafe_allow_html=True)

    st.divider()

    # ── Key Metrics ────────────────────────────────────────────────────────
    st.markdown('<span class="lbl">Key Fundamentals</span>',unsafe_allow_html=True)
    km=st.columns(6)
    km[0].metric("P/E TTM",f"{m['pe_ttm']:.1f}x" if m.get("pe_ttm") else "—",
        help="Price/Earnings. Lower = cheaper. Compare vs sector.")
    km[1].metric("Fwd P/E",f"{m['pe_fwd']:.1f}x" if m.get("pe_fwd") else "—",
        help="Based on next year's earnings estimates.")
    km[2].metric("ROE",f"{m['roe']:.1f}%" if m.get("roe") else "—",
        help=">15% = good. >25% = excellent. Best indicator of business quality.")
    km[3].metric("Op Margin",f"{m['op_margin']:.1f}%" if m.get("op_margin") else "—",
        help="Operating profit / Revenue. Pricing power + efficiency.")
    km[4].metric("D/E Ratio",f"{m['de_ratio']:.0f}%" if m.get("de_ratio") is not None else "—",
        help="Debt/Equity. <100 = conservative. Yahoo reports as %.")
    km[5].metric("ROCE",f"{m['roce']:.1f}%" if m.get("roce") else "—",
        help=">18% = capital efficient. Compare vs cost of capital ~10%.")

    st.divider()

    # ── Sector Alpha + CG-Proxy + EQ-Score + Peer + Health ───────────────
    st.divider()
    alpha=compute_sector_alpha_score(m)
    st.markdown("### Sector Alpha Score™")
    sa1,sa2=st.columns([1,2])
    with sa1:
        if alpha.get("score") is not None:
            st.markdown(
                f'<div class="gauge-wrap">'
                f'<div class="gauge-score" style="color:{alpha["color"]}">{alpha["score"]:.0f}</div>'
                f'<div class="gauge-grade" style="color:{alpha["color"]}">Grade {alpha["grade"]} · Coverage {alpha.get("coverage",0):.0f}%</div>'
                f'<div class="gauge-label">{alpha["model_name"]}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.info("Not enough data for Sector Alpha Score.")
    with sa2:
        st.markdown(f'<div class="flag-y"><b>Model logic:</b> {alpha.get("model_logic","")}<br><b>Not captured in this yfinance demo:</b> {alpha.get("missing_true_metrics","")}</div>', unsafe_allow_html=True)
        comp_rows=[]
        for c in alpha.get("components",[])[:8]:
            val=c.get("value")
            comp_rows.append({"Component":c["label"],"Value":round(val,2) if isinstance(val,(int,float)) else val,"Component score":round(c["score"],0),"Weight":c["weight"]})
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, height=260)

    left,right=st.columns([1,1])

    with left:
        st.markdown("### CG-Proxy™ & EQ-Score™")
        st.caption("Preliminary governance-risk proxy + earnings-quality proxy. yfinance demo data can be incomplete.")

        with st.spinner("Computing governance & earnings scores…"):
            cg=compute_cg_score(m,t_obj)
            eq=compute_eq_score(t_obj)

        gc,ec=st.columns(2)
        with gc:
            if cg["score"] is not None:
                st.markdown(
                    f'<div class="gauge-wrap">'
                    f'<div class="gauge-score" style="color:{cg["color"]}">{cg["score"]:.0f}</div>'
                    f'<div class="gauge-grade" style="color:{cg["color"]}">Grade {cg["grade"]} · Data {cg.get("coverage",0):.0f}%</div>'
                    f'<div class="gauge-label">CG-Proxy™ · Governance Risk Proxy</div>'
                    f'<div style="font-size:11px;color:#4b5675;margin-top:8px">Available-signal model: risk fields · insider/management holding proxy · dividend · debt trajectory · cash conversion</div>'
                    f'</div>',unsafe_allow_html=True)
            else:
                st.info("Not enough data for CG-Proxy.")
        with ec:
            if eq["score"] is not None:
                st.markdown(
                    f'<div class="gauge-wrap">'
                    f'<div class="gauge-score" style="color:{eq["color"]}">{eq["score"]:.0f}</div>'
                    f'<div class="gauge-grade" style="color:{eq["color"]}">Grade {eq["grade"]} · Data {eq.get("coverage",0):.0f}%</div>'
                    f'<div class="gauge-label">EQ-Score™ · Earnings Quality</div>'
                    f'<div style="font-size:11px;color:#4b5675;margin-top:8px">CFO/NI ratio · accruals · gross margin trend · receivables quality</div>'
                    f'</div>',unsafe_allow_html=True)
            else:
                st.info("Not enough financial data for EQ-Score.")

        # CG details
        st.markdown('<span class="lbl" style="margin-top:12px">Governance Signals</span>',unsafe_allow_html=True)
        for kind,text in (cg["details"]+eq.get("details",[])):
            cls={"g":"flag-g","r":"flag-r","y":"flag-y"}.get(kind,"flag-y")
            st.markdown(f'<div class="{cls}">{text}</div>',unsafe_allow_html=True)

    with right:
        # ── Health Check ─────────────────────────────────────────────────
        st.markdown("### Health Check")
        flags,oks=[],[]
        is_fin=_is_financial_sector(m.get("sector"),m.get("industry"))
        if is_fin:
            flags.append("ℹ️ Financial-sector company: D/E, current ratio and margins need sector-specific interpretation.")
        de=m.get("de_ratio")
        if de is not None and not is_fin:
            if de>200: flags.append(f"⚠️ Very high debt — D/E {de:.0f}%")
            elif de>100: flags.append(f"🟡 Elevated debt — D/E {de:.0f}% — monitor refinancing risk")
            else: oks.append(f"Low/manageable debt (D/E {de:.0f}%)")
        cr=m.get("current_ratio")
        if cr is not None and not is_fin:
            if cr<1.0: flags.append(f"⚠️ Liquidity risk — current ratio {cr:.1f} (<1)")
            elif cr>=1.5: oks.append(f"Good liquidity (current ratio {cr:.1f})")
        roe=m.get("roe")
        if roe is not None:
            if roe<0: flags.append(f"⚠️ Negative ROE ({roe:.1f}%) — destroying shareholder value")
            elif roe<8: flags.append(f"🟡 Weak ROE ({roe:.1f}%) — barely covering cost of equity")
            elif roe>20: oks.append(f"Excellent ROE ({roe:.1f}%) — strong value creation")
            elif roe>15: oks.append(f"Strong ROE ({roe:.1f}%)")
        rev=m.get("rev_growth_ttm")
        if rev is not None:
            if rev<-10: flags.append(f"⚠️ Revenue declining sharply ({rev:.1f}% YoY)")
            elif rev<0: flags.append(f"🟡 Revenue shrinking ({rev:.1f}% YoY)")
            elif rev>20: oks.append(f"Strong revenue growth ({rev:.1f}% YoY)")
            elif rev>10: oks.append(f"Healthy revenue growth ({rev:.1f}% YoY)")
        nm=m.get("net_margin")
        if nm is not None:
            if nm<0: flags.append(f"⚠️ Negative net margins ({nm:.1f}%) — loss-making")
            elif nm>20: oks.append(f"Excellent net margins ({nm:.1f}%)")
        if fh is not None and fh<-40:
            flags.append(f"⚠️ Down {abs(fh):.0f}% from 52W high — investigate fundamental reason")
        beta=m.get("beta")
        if beta:
            if beta>2: flags.append(f"🟡 Very high volatility — beta {beta:.1f}")
            elif beta<0.6: oks.append(f"Low market sensitivity (beta {beta:.1f})")
        if m.get("div_yield") and m["div_yield"]>1:
            oks.append(f"Consistent dividend ({m['div_yield']:.1f}% yield)")
        if m.get("fcf_yield") and m["fcf_yield"]>3:
            oks.append(f"Attractive free cash flow yield ({m['fcf_yield']:.1f}%)")
        if not flags and not oks:
            st.info("Insufficient data for automated health check.")
        for f in flags[:6]: st.markdown(f'<div class="flag-r">{f}</div>',unsafe_allow_html=True)
        for o in oks[:6]: st.markdown(f'<div class="flag-g">{o}</div>',unsafe_allow_html=True)

        # ── Peer Comparison ───────────────────────────────────────────────
        st.divider()
        st.markdown("### Peer Comparison")
        peers=[]
        for sec,sec_peers in SECTOR_PEERS.items():
            if ticker in sec_peers:
                peers=[t for t in sec_peers if t!=ticker][:5]; break
        if peers:
            with st.spinner("Loading peers…"):
                peer_m=[fetch_metrics(p) for p in peers]
            all_m=[m]+[pm for pm in peer_m if not pm.get("error")]
            prows=[{
                "":"→" if pm["ticker"]==ticker else "",
                "Stock":(pm.get("name",""))[:18],
                "P/E":pm.get("pe_ttm"),"ROE":pm.get("roe"),
                "Op Mg":pm.get("op_margin"),"Rev Gr":pm.get("rev_growth_ttm"),
                "D/E":pm.get("de_ratio"),"CG-Proxy":pm.get("cg_score"),"Data":pm.get("data_quality"),
            } for pm in all_m]
            pf=pd.DataFrame(prows)
            def hl(row): return ["background:#0a1528;font-weight:700"]*len(row) if row[""]=="→" else [""]*len(row)
            def sc(v): return "color:#22c55e" if isinstance(v,(int,float)) and v>0 else "color:#ef4444" if isinstance(v,(int,float)) and v<0 else ""
            st.dataframe(pf.style.apply(hl,axis=1).map(sc,subset=["Rev Gr"])
                .format({c:lambda v:"—" if not isinstance(v,(int,float)) else f"{v:.1f}"
                         for c in ["P/E","ROE","Op Mg","Rev Gr","D/E","CG-Proxy","Data"]}),
                use_container_width=True,height=220)

    # ── Valuation Block ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="lbl">Valuation & Growth</span>',unsafe_allow_html=True)
    vc=st.columns(6)
    vc[0].metric("EV/EBITDA",f"{m['ev_ebitda']:.1f}x" if m.get("ev_ebitda") else "—",
        help="Enterprise value / EBITDA. Accounts for debt. <10x generally attractive.")
    vc[1].metric("P/B Ratio",f"{m['pb']:.2f}x" if m.get("pb") else "—",
        help="Price to book value. High P/B OK if ROE is high.")
    vc[2].metric("Graham №",f"₹{m['graham_number']:.0f}" if m.get("graham_number") else "—",
        delta=f"{m['graham_mos']:+.0f}% MOS" if m.get("graham_mos") else None,
        help="√(22.5 × EPS × BVPS). Graham's fair value ceiling.")
    vc[3].metric("FCF Yield",f"{m['fcf_yield']:.1f}%" if m.get("fcf_yield") else "—",
        help="Free cash flow / Market cap. >5% = attractive.")
    vc[4].metric("Rev Growth",f"{m['rev_growth_ttm']:+.1f}%" if m.get("rev_growth_ttm") else "—")
    vc[5].metric("Earn Growth",f"{m['earn_growth_ttm']:+.1f}%" if m.get("earn_growth_ttm") else "—")

    # ── AI Research Brief ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### AI Research Brief")
    st.caption("Data-grounded memo. It uses only the metrics already loaded in this app; it is not a recommendation.")
    key=st.session_state.api_key; prov=st.session_state.api_provider
    if not key:
        st.markdown('<div class="flag-y">🔑 Add your Groq or Anthropic API key in the top bar to generate an AI research brief.</div>',
            unsafe_allow_html=True)
    else:
        if st.button("🤖 Generate Research Memo",type="primary"):
            with st.spinner("Generating a deeper data-grounded memo…"):
                cg_txt=f"CG-Proxy™: {cg['score']}/100 (Grade {cg['grade']}, coverage {cg.get('coverage',0):.0f}%)" if cg and cg.get("score") is not None else "CG-Proxy™: unavailable or low coverage"
                eq_txt=f"EQ-Score™: {eq['score']}/100 (Grade {eq['grade']})" if eq and eq.get("score") else "EQ-Score™: unavailable/insufficient components"
                alpha_components = "; ".join([f"{c['label']}={c.get('value')} → component score {round(c.get('score',0),0)}/100, weight {c.get('weight')}" for c in alpha.get("components",[])[:8]])
                tech_txt = (
                    f"RSI {tech.get('rsi')} ({tech.get('rsi_sig')}), MACD {tech.get('macd_sig')} hist {tech.get('macd_hist')}, "
                    f"SMA trend {tech.get('sma_sig')}, Bollinger %B {tech.get('bb_pct')} ({tech.get('bb_sig')}), "
                    f"period return visible on chart: {'available' if len(prices)>=2 else 'unavailable'}"
                ) if tech else "Technical signals unavailable: fewer than 50 price observations in selected window."
                price_txt = f"₹{price:,.2f}" if price else "—"
                prompt=f"""You are preparing a research memo for a retail Indian equity research dashboard.

STRICT DATA BOUNDARY:
Use only the data below. Do not add news, management commentary, market share, order book, competitive advantages, SEBI/regulatory events, product pipeline, or macro claims unless present below. If a relevant fact is missing, write "not available in this demo data".

COMPANY SNAPSHOT:
Company: {name}
Ticker: {ticker}
Sector: {sector}
Industry: {m.get("industry","N/A")}
Market cap: {_cap(m.get("market_cap"))}
Current price: {price_txt}
Data availability: {m.get('data_quality')}%

SECTOR-AWARE SCORE:
Sector Alpha Score™: {alpha.get('score')}/100, Grade {alpha.get('grade')}, Coverage {alpha.get('coverage')}%
Sector model used: {alpha.get('model_name')}
Model logic: {alpha.get('model_logic')}
Score components: {alpha_components or 'No components available'}
Metrics not captured in this yfinance demo: {alpha.get('missing_true_metrics')}

VALUATION:
P/E TTM: {m.get("pe_ttm")}
Forward P/E: {m.get("pe_fwd")}
P/B: {m.get("pb")}
EV/EBITDA: {m.get("ev_ebitda")}
Graham Number: ₹{m.get("graham_number")}
Graham Margin of Safety: {m.get("graham_mos")}%
Analyst target: ₹{m.get("analyst_target")}
Analyst upside: {m.get("analyst_upside")}%
Analyst count: {m.get("analyst_count")}

QUALITY / PROFITABILITY:
ROE: {m.get("roe")}%
ROA: {m.get("roa")}%
ROCE: {m.get("roce")}%
Gross margin: {m.get("gross_margin")}%
Operating margin: {m.get("op_margin")}%
Net margin: {m.get("net_margin")}%
FCF yield: {m.get("fcf_yield")}%

GROWTH:
Revenue growth: {m.get("rev_growth_ttm")}%
Earnings growth: {m.get("earn_growth_ttm")}%

BALANCE SHEET / RISK:
Debt/equity: {m.get("de_ratio")}
Current ratio: {m.get("current_ratio")}
Beta: {m.get("beta")}
52-week high: ₹{m.get("year_high")}
52-week low: ₹{m.get("year_low")}
Distance from 52-week high: {m.get("from_52w_high")}%
Insider/management holding proxy: {m.get("insider_pct")}%
Institutional holding proxy: {m.get("inst_pct")}%

GOVERNANCE / EARNINGS QUALITY PROXIES:
{cg_txt}
{eq_txt}

TECHNICAL SNAPSHOT:
{tech_txt}

Write the output in this exact structure. Be specific and numeric. Avoid generic filler.

## 1) One-line view
One sentence that says what the supplied data suggests, with a confidence qualifier based on data availability and score coverage. No buy/sell/hold language.

## 2) What looks good in the data
- 3 to 5 bullets, each with at least one number from the supplied data.

## 3) What worries me / what may be weak
- 3 to 5 bullets. Include valuation risk, quality risk, leverage/liquidity risk, growth risk, technical risk, or missing-data risk where relevant.

## 4) Valuation read
Compare P/E, forward P/E, P/B, EV/EBITDA, FCF yield, Graham MOS and analyst upside where available. Say when a metric is not meaningful or unavailable.

## 5) Sector-model interpretation
Explain why the Sector Alpha model used for this sector is appropriate, what it captures well, and what it misses because this is a yfinance demo.

## 6) Technical and risk setup
Interpret RSI, MACD, SMA trend, Bollinger position, beta and 52-week drawdown using only supplied values.

## 7) What to verify manually before relying on this
- 5 concrete checks. Examples: latest annual report, quarterly result notes, promoter pledge/shareholding pattern, auditor remarks, segment growth, bank asset quality, order book, USFDA observations — but only include checks relevant to the sector/model.

## 8) Final research note
2 to 3 sentences. No target price, no buy/sell/hold, no advice. State whether this is a high-quality candidate, watchlist candidate, or insufficient-data case based only on supplied data."""
                sys_msg=(
                    "You are a senior Indian equity research assistant writing for serious retail investors. "
                    "Your job is to turn supplied metrics into a rigorous, balanced research memo. "
                    "Use only the supplied data. Do not invent facts, moat narratives, news, regulatory events, management commentary, price targets, or recommendations. "
                    "When data is missing, explicitly say it is missing. Prefer numbers over adjectives. "
                    "Avoid generic statements like 'strong fundamentals' unless immediately supported by numbers. "
                    "Do not use buy/sell/hold language. Do not provide personalized financial advice."
                )
                thesis=call_llm(prompt,sys_msg,key,prov,purpose="research",max_tokens=1900,temperature=0.2)
                if thesis:
                    st.markdown(thesis)

    # ── Add to portfolio ────────────────────────────────────────────────────
    st.divider()
    if st.button(f"➕ Add {ticker.replace('.NS','')} to Portfolio Lab"):
        if ticker not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(ticker)
        st.session_state.portfolio_data=None
        _go_to("portfolio")


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: PORTFOLIO LAB
# ╚══════════════════════════════════════════════════════════════════╝
def page_portfolio_lab():
    st.markdown("## 💼 Portfolio Lab")
    st.caption("Build a portfolio, score its health, find the optimal allocation, and plan your SIP.")

    # ── Presets
    st.markdown('<span class="lbl">Quick Load</span>',unsafe_allow_html=True)
    pc=st.columns(4)
    presets={
        "🇮🇳 Nifty Diversified":["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS","SBIN.NS","ITC.NS"],
        "💻 IT Basket":["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","LTIM.NS","PERSISTENT.NS"],
        "🏦 Banking Basket":["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","AXISBANK.NS","BAJFINANCE.NS"],
        "🏭 Capex & Defence":["LT.NS","HAL.NS","BEL.NS","SIEMENS.NS","ABB.NS","POLYCAB.NS","CGPOWER.NS"],
    }
    for col,(label,tks) in zip(pc,presets.items()):
        with col:
            if st.button(label,use_container_width=True):
                st.session_state.portfolio_tickers=tks[:]; st.session_state.portfolio_data=None; st.session_state.portfolio_opt=None

    # ── Add ticker
    st.markdown('<span class="lbl" style="margin-top:12px">Add Stocks</span>',unsafe_allow_html=True)
    ac,bc,cc=st.columns([5,2,2])
    with ac:
        new_t=st.text_input("",placeholder="RELIANCE.NS · TCS.NS…",label_visibility="collapsed",key="port_add").strip().upper()
    with bc:
        if new_t and st.button("➕ Add",type="primary",use_container_width=True):
            if new_t not in st.session_state.portfolio_tickers:
                st.session_state.portfolio_tickers.append(new_t); st.session_state.portfolio_data=None
    with cc:
        if st.button("Clear All",use_container_width=True):
            st.session_state.portfolio_tickers=[]; st.session_state.portfolio_data=None; st.session_state.portfolio_opt=None

    tickers=st.session_state.portfolio_tickers
    if not tickers:
        st.info("Add stocks above or load a preset to get started."); return

    tags=" ".join(f'<span class="pill pill-b">{t.replace(".NS","")}</span>' for t in tickers)
    st.markdown(f'<div style="margin:8px 0 12px">{tags}</div>',unsafe_allow_html=True)

    # ── Weights
    n=len(tickers)
    st.markdown('<span class="lbl">Portfolio Weights (% — auto-normalised)</span>',unsafe_allow_html=True)
    w_cols=st.columns(min(n,8)); raw_w={}
    for i,t in enumerate(tickers):
        with w_cols[i%8]:
            raw_w[t]=st.number_input(t.replace(".NS",""),min_value=0.0,max_value=100.0,
                value=round(100.0/n,0),step=5.0,format="%.0f",key=f"pw_{t}")

    ws=sum(raw_w.values())
    wn={t:v/ws for t,v in raw_w.items()} if ws>0 else {t:1/n for t in tickers}
    wc="#22c55e" if abs(ws-100)<0.5 else "#f59e0b"
    st.markdown(f'<span style="font-size:12px;color:{wc}">Total: {ws:.0f}%</span>',unsafe_allow_html=True)

    # ── Analyze
    ab,_=st.columns([3,7])
    with ab:
        if st.button("📊 Analyze Portfolio",type="primary",use_container_width=True):
            end=datetime.now(); start=end-timedelta(days=365*3)
            with st.spinner("Fetching 3Y price history…"):
                cl=fetch_close(tickers,start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
            st.session_state.portfolio_data=cl; st.session_state.portfolio_opt=None; st.rerun()

    if st.session_state.portfolio_data is None: return

    close=st.session_state.portfolio_data
    valid=[t for t in tickers if t in close.columns]
    if len(valid)<2:
        st.error("Need at least 2 stocks with price data. Check ticker symbols."); return

    w=pd.Series({t:wn.get(t,0) for t in valid}); w/=w.sum()
    rets=close[valid].pct_change().dropna()
    port_rets=(rets*w).sum(axis=1)
    port_cum=(1+port_rets).cumprod()*100

    rf=0.065; ann_ret=float((1+port_rets.mean())**ANN-1)
    ann_vol=float(port_rets.std()*np.sqrt(ANN))
    sharpe=(ann_ret-rf)/ann_vol if ann_vol>EPS else np.nan
    neg_r=port_rets[port_rets<0]
    dv=neg_r.std()*np.sqrt(ANN) if len(neg_r)>1 else np.nan
    sortino=(ann_ret-rf)/dv if dv and dv>EPS else np.nan
    cum=(1+port_rets).cumprod(); mdd=float((cum/cum.cummax()-1).min())
    tot_ret=port_cum.iloc[-1]-100

    # ── Portfolio Health Score
    with st.spinner("Computing portfolio health…"):
        metrics_list=[fetch_metrics(t) for t in valid]
    health=compute_portfolio_health(valid,wn,close,metrics_list)

    st.divider()
    h_col,m_cols=st.columns([2,5])
    with h_col:
        st.markdown(
            f'<div class="gauge-wrap">'
            f'<div class="gauge-score" style="color:{health["color"]}">{health["total"]:.0f}</div>'
            f'<div class="gauge-grade" style="color:{health["color"]}">Grade {health["grade"]}</div>'
            f'<div class="gauge-label">Portfolio Health Score</div>'
            f'</div>',unsafe_allow_html=True)

        for dim,data in health["breakdown"].items():
            pct=data["score"]/data["max"]*100
            bc="#22c55e" if pct>=70 else "#f59e0b" if pct>=45 else "#ef4444"
            st.markdown(f'<div style="margin-top:6px"><span class="lbl">{dim} ({data["score"]:.0f}/{data["max"]})</span>'
                f'<div style="background:#1a1e35;border-radius:4px;height:5px;margin-bottom:4px">'
                f'<div style="background:{bc};width:{pct:.0f}%;height:5px;border-radius:4px"></div></div>'
                f'<div style="font-size:11px;color:#4b5675">{data["detail"]}</div></div>',
                unsafe_allow_html=True)

    with m_cols:
        rm=st.columns(6)
        rm[0].metric("3Y Total Return",f"{tot_ret:+.1f}%")
        rm[1].metric("Ann. Return",f"{ann_ret*100:+.1f}%")
        rm[2].metric("Sharpe Ratio",f"{sharpe:.2f}" if not np.isnan(sharpe) else "—",
            delta="Good" if not np.isnan(sharpe) and sharpe>=1 else None)
        rm[3].metric("Sortino",f"{sortino:.2f}" if not np.isnan(sortino) else "—",
            help="Like Sharpe but only penalises downside volatility.")
        rm[4].metric("Ann. Volatility",f"{ann_vol*100:.1f}%")
        rm[5].metric("Max Drawdown",f"{mdd*100:.1f}%")

    st.divider()

    # ── Charts
    ch1,ch2=st.columns([3,2])
    with ch1:
        chg=port_cum.iloc[-1]-100; lc="#22c55e" if chg>=0 else "#ef4444"
        fig=go.Figure(go.Scatter(x=port_cum.index,y=port_cum.round(2),mode="lines",
            line=dict(color=lc,width=2),fill="tozeroy",
            fillcolor=f"rgba({'34,197,94' if chg>=0 else '239,68,68'},0.06)",
            hovertemplate="Value: %{y:.1f}<extra></extra>"))
        fig.add_hline(y=100,line_dash="dot",line_color="#1a1e35")
        _dark(fig,h=260,title="Portfolio Cumulative Return (Base=100)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with ch2:
        sl=[t.replace(".NS","") for t in valid]
        pie=go.Figure(go.Pie(labels=sl,values=[round(w[t]*100,1) for t in valid],
            hole=0.55,textinfo="label+percent",
            marker=dict(colors=COLORS[:len(valid)],line=dict(color="#050710",width=2))))
        _dark(pie,h=260,title="Allocation")
        pie.update_layout(showlegend=False,margin=dict(l=10,r=10,t=36,b=10))
        st.plotly_chart(pie,use_container_width=True)

    # ── Correlation + Individual stats
    co1,co2=st.columns([1,1])
    with co1:
        corr=rets[valid].corr().round(2)
        fig_hm=go.Figure(go.Heatmap(z=corr.values,x=sl,y=sl,colorscale="RdBu",zmid=0,zmin=-1,zmax=1,
            text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=10),
            hovertemplate="(%{x}, %{y}): %{z:.2f}<extra></extra>"))
        _dark(fig_hm,h=280,title="Correlation Matrix")
        st.plotly_chart(fig_hm,use_container_width=True)
    with co2:
        st.markdown('<span class="lbl">Individual Performance (3Y)</span>',unsafe_allow_html=True)
        irows=[]
        for t in valid:
            s=close[t].dropna(); r=s.pct_change().dropna()
            ar=((s.iloc[-1]/s.iloc[0])**(ANN/len(s))-1)*100 if len(s)>1 else 0
            av=r.std()*np.sqrt(ANN)*100
            sh=(ar/100-rf)/(av/100) if av>0 else None
            irows.append({"Stock":t.replace(".NS",""),"Weight":f"{w[t]*100:.0f}%",
                "3Y Ret":f"{(s.iloc[-1]/s.iloc[0]-1)*100:+.0f}%",
                "Ann Vol":f"{av:.0f}%","Sharpe":round(sh,2) if sh else None})
        idf=pd.DataFrame(irows)
        def cn(v):
            if isinstance(v,str) and "+" in v: return "color:#22c55e;font-weight:600"
            if isinstance(v,str) and "-" in v: return "color:#ef4444;font-weight:600"
            if isinstance(v,(int,float)): return "color:#22c55e" if v>=1 else ("color:#fbbf24" if v>=0 else "color:#ef4444")
            return ""
        st.dataframe(idf.style.map(cn,subset=["3Y Ret","Sharpe"])
            .format({"Sharpe":lambda v:f"{v:.2f}" if isinstance(v,(int,float)) else "—"}),
            use_container_width=True,height=280)

    # ── SIP Planner
    st.divider()
    st.markdown("### 📅 SIP Planner — Wealth Projector")
    st.caption("Project wealth creation using this portfolio's historical return. Not a guarantee of future returns.")
    sp1,sp2,sp3,sp4=st.columns(4)
    with sp1: monthly=st.number_input("Monthly SIP (₹)",value=10000,step=1000,min_value=500)
    with sp2: years=st.slider("Investment Horizon (Years)",1,40,15)
    with sp3: exp_ret=st.number_input("Expected Ann. Return %",value=float(round(max(float(ann_ret*100),8.0),1)),step=0.5,min_value=0.0,max_value=50.0,
        help="Assumed annual return for projection. Defaults to the portfolio's historical annualized return, floored at 8%.")
    with sp4: step_up=st.number_input("Annual Step-Up %",value=10.0,step=5.0,min_value=0.0,
        help="Increase SIP by this % each year (step-up SIP)")

    r_m=(1+exp_ret/100)**(1/12)-1
    total_inv=0; corpus=0; history=[]
    m_sip=monthly
    for y in range(years):
        for month in range(12):
            corpus=(corpus+m_sip)*(1+r_m)
            total_inv+=m_sip
            history.append({"Month":y*12+month+1,"Corpus":round(corpus),"Invested":round(total_inv)})
        m_sip*=(1+step_up/100)

    gain=corpus-total_inv; xirr_approx=exp_ret
    sc1,sc2,sc3,sc4=st.columns(4)
    sc1.metric("Final Corpus",f"₹{corpus/1e5:.1f}L" if corpus<1e7 else f"₹{corpus/1e7:.2f}Cr")
    sc2.metric("Total Invested",f"₹{total_inv/1e5:.1f}L" if total_inv<1e7 else f"₹{total_inv/1e7:.2f}Cr")
    sc3.metric("Wealth Gain",f"₹{gain/1e5:.1f}L" if gain<1e7 else f"₹{gain/1e7:.2f}Cr",
        delta=f"{gain/total_inv*100:.0f}% gain")
    sc4.metric("Wealth Multiple",f"{corpus/total_inv:.1f}x")

    sip_df=pd.DataFrame(history)
    fig_s=go.Figure()
    fig_s.add_trace(go.Scatter(x=sip_df["Month"]/12,y=sip_df["Corpus"],name="Corpus",
        fill="tozeroy",fillcolor="rgba(37,99,235,0.10)",line=dict(color="#3b82f6",width=2)))
    fig_s.add_trace(go.Scatter(x=sip_df["Month"]/12,y=sip_df["Invested"],name="Amount Invested",
        line=dict(color="#6b7280",width=1.5,dash="dot")))
    _dark(fig_s,h=260,title=f"SIP Wealth Projection · ₹{monthly:,}/month · {exp_ret}% · {years}Y")
    fig_s.update_layout(xaxis_title="Years",yaxis_title="₹",
        legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig_s,use_container_width=True)

    # ── Markowitz Optimizer
    st.divider()
    st.markdown("### 🎯 Markowitz Optimizer")
    st.caption("Maximise U = E(R) − λ·σ². Risk tolerance (λ) controls return vs. risk tradeoff.")
    oc1,oc2,oc3=st.columns(3)
    with oc1:
        risk_l=st.select_slider("Risk Tolerance",
            options=["Very Conservative","Conservative","Balanced","Growth","Aggressive"],
            value="Balanced")
        lam_m={"Very Conservative":10.0,"Conservative":5.0,"Balanced":3.0,"Growth":1.0,"Aggressive":0.3}
        lam=lam_m[risk_l]
    with oc2:
        rf_inp=st.number_input("Risk-Free Rate % (10Y G-Sec ≈ 6.8%)",value=6.8,min_value=0.0,max_value=15.0,step=0.25)
    with oc3:
        budget=st.number_input("Investment Amount (₹)",value=100000,step=10000,format="%d")

    if st.button("⚡ Optimize",type="primary"):
        with st.spinner("Running Markowitz optimization…"):
            try:
                rf_v=rf_inp/100; mu=rets[valid].mean()*ANN; cov=rets[valid].cov()*ANN; nv=len(valid)
                def neg_util(ww):
                    er=float(ww@mu.values); var=float(max(ww@cov.values@ww,0))
                    return -(er-lam*var)
                bounds=[(0,1)]*nv; cons=[{"type":"eq","fun":lambda ww:ww.sum()-1}]
                rng=np.random.default_rng(42)
                starts=[np.ones(nv)/nv]+[rng.dirichlet(np.ones(nv)) for _ in range(9)]
                best=None
                for s0 in starts:
                    res=minimize(neg_util,s0,method="SLSQP",bounds=bounds,constraints=cons,
                        options={"maxiter":1000,"ftol":1e-12})
                    if best is None or res.fun<best.fun: best=res
                opt_w=np.maximum(best.x,0); opt_w/=opt_w.sum()
                opt_s=pd.Series(opt_w,index=valid)
                opt_r=(rets[valid]@opt_s)
                opt_ar=float((1+opt_r.mean())**ANN-1)
                opt_vol=float(opt_r.std()*np.sqrt(ANN))
                opt_sh=(opt_ar-rf_v)/opt_vol if opt_vol>EPS else np.nan
                int_alloc=cash_rem=None
                if budget>0:
                    px=close[valid].iloc[-1].values
                    shares=np.floor(opt_w*budget/px).astype(int)
                    invested=shares*px; cash_rem=float(budget-invested.sum())
                    int_alloc=pd.DataFrame({
                        "Opt Weight":[f"{v*100:.1f}%" for v in opt_w],
                        "Price (₹)":px.round(0),"Shares":shares,"Amount (₹)":invested.round(0)
                    },index=[t.replace(".NS","") for t in valid])
                st.session_state.portfolio_opt={"w":opt_s,"ar":opt_ar,"vol":opt_vol,"sh":opt_sh,"int_alloc":int_alloc,"cash":cash_rem}
                st.rerun()
            except Exception as e:
                st.error(f"Optimization error: {e}")

    if st.session_state.portfolio_opt:
        opt=st.session_state.portfolio_opt; opt_w=opt["w"]
        cmp1,cmp2=st.columns(2)
        with cmp1:
            st.markdown("**Your Weights**")
            st.metric("Ann. Return",f"{ann_ret*100:+.1f}%")
            st.metric("Sharpe",f"{sharpe:.2f}" if not np.isnan(sharpe) else "—")
            st.metric("Volatility",f"{ann_vol*100:.1f}%")
        with cmp2:
            dr=(opt["ar"]-ann_ret)*100
            ds=opt["sh"]-sharpe if not (np.isnan(sharpe) or np.isnan(opt["sh"])) else None
            st.markdown("**Optimised Weights**")
            st.metric("Ann. Return",f"{opt['ar']*100:+.1f}%",f"{dr:+.1f}%")
            st.metric("Sharpe",f"{opt['sh']:.2f}" if not np.isnan(opt["sh"]) else "—",
                delta=f"{ds:+.2f}" if ds else None)
            st.metric("Volatility",f"{opt['vol']*100:.1f}%")

        cur_wts=[w[t]*100 for t in valid]; opt_wts=[opt_w[t]*100 for t in valid]
        fig_b=go.Figure()
        fig_b.add_trace(go.Bar(name="Your %",x=sl,y=cur_wts,marker_color="#374151",opacity=0.85))
        fig_b.add_trace(go.Bar(name="Optimised %",x=sl,y=opt_wts,marker_color="#3b82f6"))
        _dark(fig_b,h=260,title="Weight Comparison")
        fig_b.update_layout(barmode="group",yaxis_title="Weight %",legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig_b,use_container_width=True)

        if opt["int_alloc"] is not None:
            st.markdown('<span class="lbl">Share-Level Allocation</span>',unsafe_allow_html=True)
            ia1,ia2=st.columns(2)
            ia1.metric("Total Invested",f"₹{opt['int_alloc']['Amount (₹)'].sum():,.0f}")
            ia2.metric("Cash Remaining",f"₹{opt['cash']:,.0f}")
            st.dataframe(opt["int_alloc"].style.format({"Price (₹)":"₹{:,.0f}","Amount (₹)":"₹{:,.0f}"}),
                use_container_width=True)

        st.download_button("⬇️ Export Allocation CSV",
            data=(opt["int_alloc"] if opt["int_alloc"] is not None
                  else pd.DataFrame({"ticker":valid,"weight":[f"{opt_w[t]*100:.1f}%" for t in valid]})).to_csv(),
            file_name="optimized_portfolio.csv",mime="text/csv")


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: RED FLAG RADAR
# ╚══════════════════════════════════════════════════════════════════╝
def page_red_flag_radar():
    st.markdown("## 🚨 Risk Radar")
    st.caption(
        "Preliminary scan for visible governance-proxy and earnings-quality risk signals. "
        "This is a triage tool, not a fraud detector or investment recommendation."
    )

    st.markdown("""
    <div style="background:#0c0e1c;border:1px solid #1a1e35;border-radius:12px;padding:16px 20px;margin-bottom:16px">
    <div style="font-size:13px;color:#9ca3af;line-height:1.7">
    <strong style="color:#e2e6ff">What Risk Radar scans for:</strong><br>
    🔴 <strong>Governance-proxy risk</strong> — available audit/board/compensation/shareholder-rights risk fields where yfinance provides them<br>
    🔴 <strong>Low insider/management holding proxy</strong> — not the same as Indian promoter holding; verify shareholding pattern manually<br>
    🟡 <strong>Debt Expansion</strong> — Debt growing faster than revenue (balance sheet stress signal)<br>
    🟡 <strong>Weak Cash Conversion</strong> — Reported profits not converting to operating cash flow<br>
    🔴 <strong>Revenue Decline</strong> — Negative revenue growth (business contraction)<br>
    🔴 <strong>Negative ROE</strong> — Destroying shareholder value
    </div>
    </div>
    """,unsafe_allow_html=True)

    cache_age=None
    if st.session_state.radar_ts:
        cache_age=(datetime.now()-st.session_state.radar_ts).seconds/3600
    if st.session_state.radar_results is not None and cache_age and cache_age<6:
        st.markdown(f'<span class="lbl">Cached results ({cache_age:.1f}h ago) · {len(st.session_state.radar_results)} flagged companies</span>',
            unsafe_allow_html=True)
        _render_radar(st.session_state.radar_results)
        return

    cols=st.columns(3)
    with cols[0]: min_flags=st.slider("Min risk signals to show",1,5,2)
    with cols[1]: sector_f=st.selectbox("Sector filter",["All"]+list(SECTOR_PEERS.keys()))
    with cols[2]: radar_universe=st.selectbox("Scan Universe",list(UNIVERSE_PRESETS.keys()),index=1,key="radar_uni")

    if not st.button("🔍 Run Risk Scan",type="primary"):
        st.info("Click 'Run Risk Scan' to scan companies. MVP scan universe is capped at Nifty 500."); return

    scan_list = _get_universe(radar_universe)

    results=[]; prog=st.progress(0.0)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(fetch_metrics,t):t for t in scan_list}
        done=0
        for fut in as_completed(futs):
            done+=1; prog.progress(done/len(scan_list),text=f"Scanning {done}/{len(scan_list)}…")
            m=fut.result()
            if m.get("error") or not m.get("name") or m["name"]==m["ticker"]: continue
            if sector_f!="All" and sector_f not in (m.get("sector","") or ""): continue

            flags=[]; warnings_list=[]

            is_fin=_is_financial_sector(m.get("sector"),m.get("industry"))
            alpha=compute_sector_alpha_score(m)
            if alpha.get("score") is not None:
                if alpha["score"]<35:
                    flags.append(f"🔴 Low Sector Alpha Score ({alpha['score']:.0f}/100) for {alpha['model_name']}")
                elif alpha["score"]<50:
                    warnings_list.append(f"🟡 Weak Sector Alpha Score ({alpha['score']:.0f}/100) for {alpha['model_name']}")

            # Governance-proxy risk
            cg=m.get("cg_score",50)
            if cg is not None:
                if cg<35: flags.append(f"🔴 Very low CG-Proxy™ ({cg:.0f}/100)")
                elif cg<50: warnings_list.append(f"🟡 Below-average governance proxy ({cg:.0f}/100)")

            ins=m.get("insider_pct")
            if ins is not None and ins<10:
                flags.append(f"🔴 Very low insider/management holding proxy ({ins:.1f}%) — verify promoter shareholding separately")
            elif ins is not None and ins<20:
                warnings_list.append(f"🟡 Low insider holding ({ins:.1f}%)")

            # Financial health
            if m.get("roe") is not None and m["roe"]<0:
                flags.append(f"🔴 Negative ROE ({m['roe']:.1f}%) — destroying value")
            if m.get("rev_growth_ttm") is not None and m["rev_growth_ttm"]<-10:
                flags.append(f"🔴 Revenue shrinking {m['rev_growth_ttm']:.1f}% YoY")
            elif m.get("rev_growth_ttm") is not None and m["rev_growth_ttm"]<0:
                warnings_list.append(f"🟡 Negative revenue growth ({m['rev_growth_ttm']:.1f}%)")
            if not is_fin and m.get("de_ratio") is not None and m["de_ratio"]>200:
                flags.append(f"🔴 Very high debt (D/E {m['de_ratio']:.0f}%) — refinancing risk")
            elif not is_fin and m.get("de_ratio") is not None and m["de_ratio"]>120:
                warnings_list.append(f"🟡 Elevated debt (D/E {m['de_ratio']:.0f}%)")
            if not is_fin and m.get("current_ratio") is not None and m["current_ratio"]<1.0:
                flags.append(f"🔴 Liquidity risk (current ratio {m['current_ratio']:.1f})")
            if is_fin:
                warnings_list.append("🟡 Financial-sector company: leverage/liquidity ratios need sector-specific manual review")
            if m.get("net_margin") is not None and m["net_margin"]<0:
                flags.append(f"🔴 Loss-making (net margin {m['net_margin']:.1f}%)")
            if m.get("from_52w_high") is not None and m["from_52w_high"]<-50:
                warnings_list.append(f"🟡 Down {abs(m['from_52w_high']):.0f}% from 52W high")

            total_signals=len(flags)+len(warnings_list)//2
            if total_signals>=min_flags or len(flags)>=1:
                m["_flags"]=flags; m["_warnings"]=warnings_list
                m["sector_score"]=alpha.get("score")
                m["sector_model"]=alpha.get("model_name")
                m["_risk_score"]=len(flags)*20+len(warnings_list)*5
                results.append(m)

    prog.empty()
    results.sort(key=lambda x:x.get("_risk_score",0),reverse=True)
    st.session_state.radar_results=results; st.session_state.radar_ts=datetime.now()
    _render_radar(results)


def _render_radar(results):
    if not results:
        st.success("✅ No major risk signals found with current filters. This still does not replace manual research."); return

    st.markdown(f'<div class="flag-r">⚠️ Found {len(results)} companies with elevated risk signals. This does NOT mean fraud or avoid — it means manual verification is warranted.</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    # Summary table
    rows=[{
        "Company":(m.get("name",""))[:22],
        "Ticker":m["ticker"].replace(".NS",""),
        "Sector":(m.get("sector","N/A") or "N/A")[:14],
        "Model":(m.get("sector_model") or "")[:22],
        "Alpha":f"{m['sector_score']:.0f}" if m.get("sector_score") is not None else "—",
        "Risk Score":m.get("_risk_score",0),
        "CG-Proxy":f"{m['cg_score']:.0f}" if m.get("cg_score") else "—",
        "ROE%":m.get("roe"),
        "Rev Gr%":m.get("rev_growth_ttm"),
        "D/E":m.get("de_ratio"),
        "Data":f"{m.get('data_quality',0):.0f}%" if m.get("data_quality") is not None else "—",
        "Risk Signals":len(m.get("_flags",[])),
        "Warnings":len(m.get("_warnings",[])),
    } for m in results[:20]]
    df=pd.DataFrame(rows)

    def risk_style(val):
        if not isinstance(val,(int,float)): return ""
        if val>=60: return "background:#1a0505;color:#f87171;font-weight:700"
        if val>=30: return "background:#1a1205;color:#fbbf24;font-weight:600"
        return ""
    def neg_col(v): return "color:#ef4444" if isinstance(v,(int,float)) and v<0 else ""

    st.dataframe(df.style.map(risk_style,subset=["Risk Score"])
        .map(neg_col,subset=["ROE%","Rev Gr%"])
        .format({"ROE%":lambda v:f"{v:.1f}" if isinstance(v,(int,float)) else "—",
                 "Rev Gr%":lambda v:f"{v:+.1f}%" if isinstance(v,(int,float)) else "—",
                 "D/E":lambda v:f"{v:.0f}%" if isinstance(v,(int,float)) else "—"}),
        use_container_width=True,height=400)

    # Detail cards
    st.divider()
    st.markdown("### Detailed Risk Signals")
    for m in results[:10]:
        name=m.get("name",m["ticker"]); flags=m.get("_flags",[]); warns=m.get("_warnings",[])
        risk=m.get("_risk_score",0); rc="#ef4444" if risk>=60 else "#f59e0b"
        with st.expander(f"**{name}** ({m['ticker'].replace('.NS','')}) — Risk Score: {risk}",expanded=False):
            mc1,mc2=st.columns([2,3])
            with mc1:
                st.metric("P/E",f"{m['pe_ttm']:.1f}x" if m.get("pe_ttm") else "—")
                st.metric("ROE",f"{m.get('roe','—'):.1f}%" if isinstance(m.get("roe"),(int,float)) else "—")
                st.metric("Rev Growth",f"{m.get('rev_growth_ttm','—'):+.1f}%" if isinstance(m.get("rev_growth_ttm"),(int,float)) else "—")
                st.metric("D/E",f"{m.get('de_ratio','—'):.0f}%" if isinstance(m.get("de_ratio"),(int,float)) else "—")
            with mc2:
                for f in flags: st.markdown(f'<div class="flag-r">{f}</div>',unsafe_allow_html=True)
                for w in warns: st.markdown(f'<div class="flag-y">{w}</div>',unsafe_allow_html=True)

            if st.button(f"🔬 Deep Dive → {name[:20]}",key=f"rd_{m['ticker']}"):
                _go_to("analyze", m["ticker"])


# ╔══════════════════════════════════════════════════════════════════╗
#  PAGE: METRICS GUIDE
# ╚══════════════════════════════════════════════════════════════════╝
def page_metrics_guide():
    st.markdown("## 📘 Metrics Guide")
    st.caption("How AlphaMagic calculates the main metrics, why each is useful, and where it can mislead.")

    st.markdown("""
    <div class="flag-y">
    This MVP uses yfinance demo data. Indian fundamentals, promoter/shareholding details, pledge data, auditor qualifications,
    SEBI actions and exchange-filings data are not fully captured here. Scores are research prompts, not investment advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Sector Alpha Score™ — how the sector-aware score works")
    st.markdown("""
    <div class="ai-box">
    <b>Sector Alpha Score™ is a 0–100 sector-aware triage score.</b><br>
    It first classifies a company into a practical model — financials, IT, healthcare, FMCG, consumer cyclical,
    industrials, energy/materials, communication/new-age, or general non-financial. It then uses different metric weights
    for that model. This prevents banks/NBFCs from being penalized using manufacturing-style D/E or current ratio, and prevents
    asset-light IT/FMCG businesses from being judged like commodity cyclicals.<br><br>
    The score is still only as good as the input data. For this demo, yfinance is used. True bank asset quality, promoter pledge,
    auditor qualifications, related-party transactions and regulatory actions need a better API/filings parser later.
    </div>
    """, unsafe_allow_html=True)

    score_rows=[]
    for key,note in SECTOR_MODEL_NOTES.items():
        score_rows.append({"Sector model":note["name"],"What the score emphasizes":note["logic"],"Important missing true metrics in demo":note["missing"]})
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, height=430)

    st.markdown("### Main metrics")
    sections = [
        ("Data Availability", "Available key fields / total key fields × 100", "Shows whether the app has enough data to trust the screen. Low data availability should reduce confidence in every other output."),
        ("Sector Alpha Score™", "Weighted 0–100 score using the sector model + current lens", "Main ranking score. Uses different weights by sector. Coverage shows how much of the score was actually computable."),
        ("P/E TTM", "Current market price / trailing 12-month EPS", "Quick valuation multiple. Useful only versus sector peers and growth quality; low P/E can be a value trap."),
        ("Forward P/E", "Current market price / expected next-year EPS", "Captures analyst expectations. It is estimate-driven and can change sharply after results."),
        ("P/B", "Market price / book value per share", "Especially useful for banks, NBFCs, insurers, asset-heavy companies and cyclicals. Less useful for asset-light compounders."),
        ("ROE", "Net income / shareholders’ equity × 100", "Measures return generated on shareholder capital. High ROE is good only if not driven by excessive leverage."),
        ("ROA", "Net income / total assets × 100", "Important for banks/NBFCs because assets are the earning base. This is a better financial-sector quality proxy than D/E."),
        ("ROCE", "EBIT / (total assets − current liabilities) × 100", "Measures return on capital employed, useful for comparing operating quality across non-financial businesses."),
        ("Operating margin", "Operating income / revenue × 100", "Shows operating efficiency and pricing power. Must be compared within the same sector."),
        ("Gross margin", "Gross profit / revenue × 100", "Useful for pharma, consumer, internet and platform companies where contribution economics matter."),
        ("Net margin", "Net income / revenue × 100", "Shows after-tax profitability. Can be distorted by one-offs or accounting gains/losses."),
        ("Revenue growth", "Latest period revenue vs prior comparable period", "Shows business expansion. For financials, this is only a rough income-growth proxy, not loan-book growth."),
        ("Earnings growth", "Latest period earnings vs prior comparable period", "Shows profit expansion. Cyclical rebounds and low bases can mislead."),
        ("D/E", "Total debt / shareholders’ equity", "Balance-sheet leverage proxy for non-financial companies. Not interpreted the same way for banks/NBFCs/insurers."),
        ("Current ratio", "Current assets / current liabilities", "Short-term liquidity proxy for non-financial firms. Weak metric for banks/NBFCs."),
        ("FCF yield", "Free cash flow / market cap × 100", "Cash return relative to market value. Higher can be attractive if cash flows are sustainable."),
        ("Graham Number", "√(22.5 × EPS × book value per share)", "Conservative value-investing ceiling. Less suitable for high-growth, asset-light or loss-making companies."),
        ("Graham MOS", "(Graham Number − price) / Graham Number × 100", "Positive value suggests price below Graham ceiling. It is not intrinsic value by itself."),
        ("Analyst upside", "(mean target price − current price) / current price × 100", "Shows consensus expectation, not certainty. Coverage can be sparse or biased."),
        ("52-week distance", "(current price − 52-week high) / 52-week high × 100", "Momentum/drawdown context. A big fall can mean opportunity or fundamental damage."),
        ("Beta", "Stock return sensitivity versus market", "Volatility proxy. Historical beta may not predict future drawdowns."),
        ("RSI 14", "Relative Strength Index over 14 periods", "Technical overbought/oversold oscillator. Not a standalone buy/sell signal."),
        ("MACD", "12-period EMA − 26-period EMA, compared with 9-period signal line", "Trend/momentum indicator. Works poorly in sideways markets."),
        ("SMA 50/200", "50-day and 200-day simple moving averages", "Trend filter; price above both often means uptrend, but it lags."),
        ("Bollinger %B", "Position of price within 20-day ±2 SD Bollinger Bands", "Shows whether price is near upper/lower volatility band. Extremes can persist."),
        ("CG-Proxy™", "Normalized score from available yfinance governance-risk fields, insider-holding proxy, dividend proxy, debt trajectory and cash conversion", "A triage proxy only. It is not promoter holding, not an audit opinion, and not a formal governance rating."),
        ("EQ-Score™", "Cash conversion + accruals + gross-margin trend + receivables-vs-revenue trend", "Checks whether reported profits look cash-backed. Score is hidden if fewer than 3/4 components are available."),
        ("Portfolio Health", "Diversification + sector-aware average quality + historical volatility/drawdown + momentum", "A simple historical diagnostic. It is not a prediction or an optimized investment plan."),
        ("Markowitz optimizer", "Historical mean-variance optimization", "Uses past returns and volatility; can overfit badly. Use with allocation caps and manual judgment."),
        ("SIP planner", "Monthly investment compounded at user-selected expected return", "Scenario calculator only. It does not forecast real returns."),
    ]
    df=pd.DataFrame(sections, columns=["Metric", "Calculation", "Why it matters / limitation"])
    st.dataframe(df, use_container_width=True, height=760)

    with st.expander("How the sector score should improve when you move beyond yfinance"):
        st.markdown("""
        - **Banks:** GNPA, NNPA, NIM, CASA, credit cost, provision coverage ratio, CRAR, loan/deposit growth.  
        - **NBFCs:** AUM growth, spreads, stage-3 assets, collection efficiency, ALM gaps, borrowing mix.  
        - **Insurance:** solvency ratio, VNB margin, persistency, combined ratio for general insurance.  
        - **IT:** constant-currency growth, EBIT margin, deal wins, attrition, client concentration.  
        - **Pharma:** USFDA risk, R&D intensity, ANDA pipeline, geography concentration, product concentration.  
        - **Industrials:** order book, working-capital days, execution cycle, receivables from government clients.  
        - **FMCG/consumer:** volume growth, gross margin, ad-spend intensity, distribution reach.  
        - **Cyclicals:** capacity utilization, commodity spreads, inventory cycle, regulated returns where applicable.
        """)


# ╔══════════════════════════════════════════════════════════════════╗
#  TOP NAVIGATION
# ╚══════════════════════════════════════════════════════════════════╝
# Apply page jumps requested by row-level buttons BEFORE creating the radio
# widget. This keeps the radio, page router and Deep Dive ticker synchronized.
_apply_pending_navigation()

nl,nc,nr=st.columns([2,7,3])
with nl:
    st.markdown('<div style="font-size:20px;font-weight:900;color:#e2e6ff;padding:4px 0">✨ AlphaMagic</div>',
        unsafe_allow_html=True)
with nc:
    if "main_nav" not in st.session_state:
        st.session_state.main_nav = PAGE_TO_NAV.get(st.session_state.get("page", "pulse"), NAV_LABELS[0])
    nav=st.radio("", NAV_LABELS,
        horizontal=True,label_visibility="collapsed",key="main_nav")
    new_page=PAGE_MAP[nav]
    if new_page!=st.session_state.page:
        st.session_state.page=new_page
with nr:
    apc1,apc2=st.columns([3,1])
    with apc1:
        k=st.text_input("",value=st.session_state.api_key,type="password",
            placeholder="Groq or Anthropic API key",label_visibility="collapsed",key="nav_key")
        if k and k!=st.session_state.api_key: st.session_state.api_key=k
    with apc2:
        prov=st.selectbox("",["Groq","Anthropic"],label_visibility="collapsed",key="nav_prov")
        if prov!=st.session_state.api_provider: st.session_state.api_provider=prov

st.markdown('<hr style="margin:8px 0 20px;border-color:#1a1e35">',unsafe_allow_html=True)

# ── Route
p=st.session_state.page
if   p=="pulse":     page_market_pulse()
elif p=="screen":    page_ai_screener()
elif p=="analyze":   page_deep_dive()
elif p=="portfolio": page_portfolio_lab()
elif p=="radar":     page_red_flag_radar()
elif p=="metrics":   page_metrics_guide()
