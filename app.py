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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.optimize import minimize   # for re-estimating DCC on an asset subset

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
        "std_resid":load("std_resid"),
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
# Re-estimate DCC(1,1) on a SUBSET of assets (1–5 of the selected stocks)
# --------------------------------------------------------------------------- #
# The univariate GARCH step is subset-invariant, so the saved standardized
# residuals are reused as-is — only the cheap correlation step is recomputed.
@st.cache_data(show_spinner="Estimating DCC on selected assets…")
def estimate_dcc(cols: tuple, lo_ts=None, hi_ts=None):
    """DCC(1,1) on `cols` using the saved standardized residuals.
    Returns a/b, the conditional-correlation series, and the matrices R_t.
    Returns None when fewer than 2 assets are given (a correlation needs a pair)."""
    sr = D["std_resid"][list(cols)].dropna()
    if lo_ts is not None and hi_ts is not None:      # optional: estimate on a window
        sr = sr.loc[lo_ts:hi_ts]
    if sr.shape[1] < 2 or len(sr) < 30:
        return None

    E = sr.values
    T, N = E.shape
    Qbar = np.cov(E, rowvar=False)

    def dcc_filter(a, b):
        Q = Qbar.copy()
        Rt = np.empty((T, N, N))
        for t in range(T):
            if t > 0:
                e = E[t - 1][:, None]
                Q = (1 - a - b) * Qbar + a * (e @ e.T) + b * Q
            d = np.sqrt(np.diag(Q))
            Rt[t] = Q / np.outer(d, d)
        return Rt

    def neg_loglik(theta):
        a, b = theta
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        Rt = dcc_filter(a, b)
        ll = 0.0
        for t in range(T):
            et = E[t][:, None]
            _, logdet = np.linalg.slogdet(Rt[t])
            ll += logdet + (et.T @ np.linalg.solve(Rt[t], et))[0, 0] - (et.T @ et)[0, 0]
        return 0.5 * ll

    opt = minimize(neg_loglik, x0=[0.02, 0.95], method="L-BFGS-B",
                   bounds=[(1e-6, 0.5), (1e-6, 0.999)])
    a, b = opt.x
    Rt = dcc_filter(a, b)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    corr = pd.DataFrame(
        {f"{cols[i]}-{cols[j]}": Rt[:, i, j] for i, j in pairs}, index=sr.index)
    return {"a": a, "b": b, "Rt": Rt, "corr": corr,
            "assets": list(cols), "loglik": -opt.fun, "n_obs": T}

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

st.sidebar.divider()
reestimate = st.sidebar.toggle(
    "Re-estimate DCC on selected assets",
    value=False,
    help="Off = show the full-sample 5-asset model from the notebook. "
         "On = re-run the DCC step on just the selected assets.",
)
est_on_window = st.sidebar.checkbox(
    "Estimate on the visible date window", value=False, disabled=not reestimate,
    help="Off = use the full sample (recommended, more stable). "
         "On = fit α, β only on the selected window (regime-specific).",
)

# Run the subset estimation when requested (cached on the asset tuple + window).
sub = None
if reestimate:
    win = (lo, hi) if est_on_window else (None, None)
    sub = estimate_dcc(tuple(sorted(sel_assets)), *win)


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

# If re-estimating, show the subset's α/β instead of the full-sample numbers.
if reestimate and sub is not None:
    a_hat, b_hat = sub["a"], sub["b"]
    src = f"{len(sub['assets'])} selected assets · {sub['n_obs']} obs"
elif reestimate and sub is None:
    src = "DCC needs ≥2 assets — pick more"
else:
    src = "full-sample model (5 assets)"

c1, c2, c3, c4 = st.columns(4)
c1.metric("DCC α", f"{a_hat:.3f}" if not (reestimate and sub is None) else "—")
c2.metric("DCC β", f"{b_hat:.3f}" if not (reestimate and sub is None) else "—")
c3.metric("Persistence α+β",
          f"{a_hat + b_hat:.3f}" if not (reestimate and sub is None) else "—")
c4.metric("Model", f"{len(sel_assets)} sel." if reestimate else len(ASSETS))
st.caption(f"Estimated on: **{src}**.")

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

    st.subheader("Bivariate scatterplots of daily returns")
    r_pair = clip(returns)[sel_assets]
    g = sns.pairplot(r_pair, kind="scatter", diag_kind="kde",
                     plot_kws=dict(s=8, alpha=0.3, edgecolor="none"),
                     corner=True, height=1.9)
    g.figure.suptitle("Bivariate scatterplots of daily returns", y=1.02)
    st.pyplot(g.figure, use_container_width=True)
    plt.close(g.figure)

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
    use_sub = reestimate and sub is not None
    src_label = ("re-estimated on selected assets" if use_sub
                 else "full-sample 5-asset model")
    st.subheader("DCC conditional correlations  ρ$_{ij,t}$")
    st.caption(f"Source: {src_label}.")

    if reestimate and sub is None:
        st.warning("Select at least 2 assets to estimate a DCC correlation.")

    # choose which correlation set + pair list to plot
    corr_src = sub["corr"] if use_sub else D["corr"]
    pair_src = list(corr_src.columns) if use_sub else sel_pairs

    if pair_src:
        cs = clip(corr_src)[[p for p in pair_src if p in corr_src.columns]]
        fig = px.line(cs, template=TEMPLATE, color_discrete_sequence=PALETTE,
                      labels={"value": "correlation", "Date": "", "variable": "pair"})
        fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(height=460, legend_title_text="", margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    elif not (reestimate and sub is None):
        st.warning("Select at least two assets to show a pair.")

    st.subheader("Correlation matrix on a chosen date")
    st.caption("Rebuilt on the fly from the pairwise series — pick any trading day.")
    heat_assets = sub["assets"] if use_sub else ASSETS
    cs_full = clip(corr_src)
    if len(cs_full) and len(heat_assets) >= 2:
        day = st.select_slider("Date", options=list(cs_full.index),
                               value=cs_full.index[-1],
                               format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
        # reconstruct the symmetric NxN correlation matrix for that day
        row = corr_src.loc[day]
        M = pd.DataFrame(np.eye(len(heat_assets)), index=heat_assets, columns=heat_assets)
        for pair, val in row.items():
            i, j = pair.split("-")
            M.loc[i, j] = M.loc[j, i] = val
        if not use_sub:
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
