"""
DCC-GARCH Dashboard — Financial Econometrics HW5
================================================
Interactive visualisation of the DCC(1,1)-GARCH(1,1) results.

Reads the parquet files written by the notebook's "save" cell (folder: ./data)
and never re-estimates anything — the heavy GARCH/DCC work stays in the notebook,
the app just loads and plots.

Run with:   streamlit run app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config & light styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="DCC-GARCH Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# A small, restrained palette + plotly template applied to every chart.
TEMPLATE = "plotly_white"
PALETTE = px.colors.qualitative.Set2
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA = Path(__file__).parent / "data"

# --------------------------------------------------------------------------- #
# Cached loaders — each parquet read happens once per session
# --------------------------------------------------------------------------- #
@st.cache_data
def load(name: str) -> pd.DataFrame:
    """Load one parquet file; the DatetimeIndex named 'Date' round-trips automatically."""
    return pd.read_parquet(DATA / f"{name}.parquet")


@st.cache_data
def load_all():
    return {
        "prices":   load("prices"),
        "returns":  load("returns"),
        "cond_vol": load("cond_vol"),
        "corr":     load("corr_series"),
        "weights":  load("mvp_weights"),
        "port":     load("portfolio_returns"),
        "desc":     load("descriptives"),
        "uncond":   load("uncond_corr"),
        "dcc":      load("dcc_params"),
        "garch":    load("garch_params"),
    }


if not DATA.exists():
    st.error(
        f"No `data/` folder found next to app.py (looked in {DATA}).\n\n"
        "Run the **save cell** in HW5.ipynb first to create the parquet files."
    )
    st.stop()

D = load_all()
returns = D["returns"]
ASSETS = list(returns.columns)
PAIRS = list(D["corr"].columns)

# --------------------------------------------------------------------------- #
# Sidebar controls (shared across tabs)
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")

sel_assets = st.sidebar.multiselect("Assets", ASSETS, default=ASSETS)
if not sel_assets:
    st.sidebar.warning("Select at least one asset.")
    sel_assets = ASSETS

dmin, dmax = returns.index.min().date(), returns.index.max().date()
date_range = st.sidebar.slider(
    "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="YYYY-MM"
)
lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])


def clip(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a date-indexed frame to the sidebar window."""
    return df.loc[lo:hi]


# pairs whose BOTH legs are in the current asset selection
sel_pairs = [p for p in PAIRS if all(leg in sel_assets for leg in p.split("-"))]

# --------------------------------------------------------------------------- #
# Header + headline model numbers
# --------------------------------------------------------------------------- #
st.title("Dynamic Conditional Correlations — DCC-GARCH")
st.caption("Financial Econometrics · HW5 · multivariate volatility & minimum-variance portfolio")

a_hat = float(D["dcc"]["a"].iloc[0])
b_hat = float(D["dcc"]["b"].iloc[0])
c1, c2, c3, c4 = st.columns(4)
c1.metric("DCC α", f"{a_hat:.3f}")
c2.metric("DCC β", f"{b_hat:.3f}")
c3.metric("Persistence α+β", f"{a_hat + b_hat:.3f}")
c4.metric("Assets", len(ASSETS))

tab_ret, tab_vol, tab_corr, tab_port = st.tabs(
    ["Returns & Stats", "Conditional Volatility", "Conditional Correlations", "Portfolio"]
)

# =========================================================================== #
# TAB 1 — Returns & descriptive statistics (Q1)
# =========================================================================== #
with tab_ret:
    st.subheader("Daily percentage log-returns")
    r = clip(returns)[sel_assets]
    fig = px.line(r, template=TEMPLATE, color_discrete_sequence=PALETTE,
                  labels={"value": "return (%)", "Date": "", "variable": "asset"})
    fig.update_layout(height=420, legend_title_text="", margin=dict(t=10, b=0))
    fig.update_traces(line=dict(width=0.7))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Descriptive statistics")
        st.dataframe(D["desc"].loc[sel_assets].style.format("{:.4f}"),
                     use_container_width=True)
    with right:
        st.subheader("Unconditional correlation")
        u = D["uncond"].loc[sel_assets, sel_assets]
        heat = px.imshow(u, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto", template=TEMPLATE)
        heat.update_layout(height=360, margin=dict(t=10, b=0), coloraxis_showscale=True)
        st.plotly_chart(heat, use_container_width=True)

    st.subheader("Cumulative price path")
    p = clip(D["prices"])[sel_assets]
    p_norm = 100 * p / p.iloc[0]            # rebased to 100 at window start
    figp = px.line(p_norm, template=TEMPLATE, color_discrete_sequence=PALETTE,
                   labels={"value": "rebased (start=100)", "Date": "", "variable": "asset"})
    figp.update_layout(height=360, legend_title_text="", margin=dict(t=10, b=0))
    st.plotly_chart(figp, use_container_width=True)

# =========================================================================== #
# TAB 2 — Conditional volatility (Q4)
# =========================================================================== #
with tab_vol:
    st.subheader("Univariate conditional standard deviations  σ$_{i,t}$  (% per day)")
    cv = clip(D["cond_vol"])[sel_assets]
    fig = px.line(cv, template=TEMPLATE, color_discrete_sequence=PALETTE,
                  labels={"value": "conditional SD (%)", "Date": "", "variable": "asset"})
    fig.update_layout(height=480, legend_title_text="", margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Synchronous spikes mark market-wide stress (e.g. the March-2020 COVID crash and the "
        "2022 tightening). Volatility is clustered and mean-reverting — the behaviour GARCH captures."
    )

# =========================================================================== #
# TAB 3 — Conditional correlations (Q5) + dynamic heatmap
# =========================================================================== #
with tab_corr:
    st.subheader("DCC conditional correlations  ρ$_{ij,t}$")
    if sel_pairs:
        cs = clip(D["corr"])[sel_pairs]
        fig = px.line(cs, template=TEMPLATE, color_discrete_sequence=PALETTE,
                      labels={"value": "correlation", "Date": "", "variable": "pair"})
        fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(height=460, legend_title_text="", margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least two assets to show a pair.")

    st.subheader("Correlation matrix on a chosen date")
    st.caption("Rebuilt on the fly from the pairwise series — pick any trading day.")
    cs_full = clip(D["corr"])
    if len(cs_full):
        day = st.select_slider("Date", options=list(cs_full.index),
                               value=cs_full.index[-1],
                               format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
        # reconstruct the symmetric NxN correlation matrix for that day
        row = D["corr"].loc[day]
        M = pd.DataFrame(np.eye(len(ASSETS)), index=ASSETS, columns=ASSETS)
        for pair, val in row.items():
            i, j = pair.split("-")
            M.loc[i, j] = M.loc[j, i] = val
        M = M.loc[sel_assets, sel_assets]
        heat = px.imshow(M, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto", template=TEMPLATE)
        heat.update_layout(height=420, margin=dict(t=10, b=0),
                           title=pd.Timestamp(day).strftime("%d %b %Y"))
        st.plotly_chart(heat, use_container_width=True)

# =========================================================================== #
# TAB 4 — Portfolio: weights (Q6), MVP returns (Q7), comparison (Q8)
# =========================================================================== #
with tab_port:
    st.subheader("Minimum-Variance Portfolio weights  w$_t^*$")
    w = clip(D["weights"])[sel_assets]
    figw = go.Figure()
    for i, col in enumerate(w.columns):
        figw.add_trace(go.Scatter(
            x=w.index, y=w[col], name=col, mode="lines",
            stackgroup="one", line=dict(width=0.4),
            fillcolor=PALETTE[i % len(PALETTE)]))
    figw.update_layout(template=TEMPLATE, height=420, legend_title_text="",
                       margin=dict(t=10, b=0), yaxis_title="weight")
    st.plotly_chart(figw, use_container_width=True)
    st.caption("Weights can be negative (short positions); the optimiser tilts toward "
               "low-volatility, weakly-correlated assets and rotates as the covariance moves.")

    st.subheader("Strategy comparison — MVP vs Equal-Weight")
    port = clip(D["port"])

    cum = port.cumsum()                      # cumulative log-returns (%)
    figc = px.line(cum, template=TEMPLATE,
                   color_discrete_map={"MVP": "#2a9d8f", "EqualWeight": "#e76f51"},
                   labels={"value": "cumulative log-return (%)", "Date": "", "variable": ""})
    figc.update_layout(height=380, margin=dict(t=10, b=0))
    st.plotly_chart(figc, use_container_width=True)

    # performance table (annualised, rf = 0)
    def stats(x: pd.Series) -> dict:
        return {
            "Mean (daily %)":   x.mean(),
            "Variance (daily)": x.var(),
            "Ann. Vol (%)":     x.std() * np.sqrt(252),
            "Ann. Return (%)":  x.mean() * 252,
            "Ann. Sharpe":      (x.mean() / x.std()) * np.sqrt(252),
        }

    comp = pd.DataFrame({c: stats(port[c]) for c in port.columns})
    m1, m2 = st.columns(2)
    m1.metric("MVP — daily variance", f"{port['MVP'].var():.4f}",
              delta=f"{port['MVP'].var() - port['EqualWeight'].var():.4f} vs EW",
              delta_color="inverse")
    m2.metric("MVP — annualised Sharpe", f"{comp.loc['Ann. Sharpe', 'MVP']:.2f}",
              delta=f"{comp.loc['Ann. Sharpe','MVP'] - comp.loc['Ann. Sharpe','EqualWeight']:+.2f} vs EW")
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

    st.info(
        "The MVP's purpose is **lower variance**; in the window above it typically delivers a higher "
        "risk-adjusted return (Sharpe) than 1/N, at the cost of turnover and possible short/levered "
        "positions. Equal-weight is costless and robust but rides the full drawdowns."
    )
