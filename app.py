"""Viking Fund Dashboard — Fundamental Analysis"""

import re
import html as html_mod
import streamlit as st
import plotly.graph_objects as go
import base64
from pathlib import Path
from datetime import datetime, timedelta
from data_manager import get_provider, _format_statement_display, fetch_earnings_calendar
import backend


def _esc(text):
    """Escape a string for safe HTML embedding."""
    if text is None:
        return ""
    return html_mod.escape(str(text))


def _sanitize_ticker(raw: str) -> str:
    """Strip to alphanumeric + dots/hyphens (valid ticker chars only)."""
    return re.sub(r"[^A-Z0-9.\-]", "", raw.upper().strip())[:10]

st.set_page_config(page_title="Viking Fund Dashboard", layout="wide")

# ── Fade-in CSS for hover tooltips ────────────────────────────────
st.markdown("""
<style>
.hoverlayer .hovertext {
    animation: fadeIn 0.2s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
/* Plotly trace legend toggle — smooth fade */
.js-plotly-plot .trace {
    transition: opacity 0.3s ease-in-out !important;
}
/* Smooth chart appearance on Streamlit re-render */
.js-plotly-plot {
    animation: fadeIn 0.3s ease-in;
}
/* Expand button — align with chart, strip grey box */
div[data-testid="stButton"] button[kind="secondary"] {
    padding: 0.15rem 0.4rem;
    min-height: 0;
    margin-top: 2.2rem;
    background: transparent;
    border: none;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

provider = get_provider()

# ── Auth gate ────────────────────────────────────────────────────────

# Encode logo early (used by auth page and sidebar)
_logo_path = Path(__file__).parent / "assets" / "logos" / "vfc_logo_transparent (3).png"
_logo_b64 = ""
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("wl_cache_v", 0)
st.session_state.setdefault("notes_cache_v", 0)
st.session_state.setdefault("wl_active_group_idx", 0)
st.session_state.setdefault("wl_dip_ma", 200)


def _render_auth_page():
    """Full-screen login / signup form."""
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    _al, _am, _ar = st.columns([1, 2, 1])
    with _am:
        if _logo_b64:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:1rem;">'
                f'<img src="data:image/png;base64,{_logo_b64}" style="width:220px;">'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<h3 style="text-align:center; margin-bottom:1.5rem;">Viking Fund Club Dashboard</h3>',
            unsafe_allow_html=True,
        )

        tab_in, tab_up = st.tabs(["Sign In", "Sign Up"])

        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    res = backend.sign_in(email, password)
                    if res["success"]:
                        st.session_state["authenticated"] = True
                        user_obj = res["user"]
                        profile = backend.get_profile(str(user_obj.id))
                        st.session_state["user"] = {
                            "id": str(user_obj.id),
                            "email": user_obj.email,
                            "first_name": profile.get("first_name", "") if profile else "",
                            "last_name": profile.get("last_name", "") if profile else "",
                            "student_id": profile.get("student_id", "") if profile else "",
                        }
                        backend.log_activity(str(user_obj.id), "login")
                        st.rerun()
                    else:
                        st.error(res["error"])

        with tab_up:
            first = st.text_input("First Name", key="signup_first")
            last = st.text_input("Last Name", key="signup_last")
            sid = st.text_input("Student ID", key="signup_sid")
            s_email = st.text_input("Email", key="signup_email")
            s_pass = st.text_input("Password", type="password", key="signup_password")
            s_pass2 = st.text_input("Confirm Password", type="password", key="signup_password2")
            if st.button("Create Account", width="stretch", type="primary"):
                if not all([first, last, sid, s_email, s_pass, s_pass2]):
                    st.error("All fields are required.")
                elif s_pass != s_pass2:
                    st.error("Passwords do not match.")
                elif len(s_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif backend.check_student_id_exists(sid):
                    st.error("This Student ID is already registered.")
                else:
                    res = backend.sign_up(s_email, s_pass, first, last, sid)
                    if res["success"]:
                        st.success("Account created! Please check your email to confirm, then sign in.")
                    else:
                        st.error(res["error"])


# Check existing session
if not st.session_state["authenticated"]:
    session = backend.get_session()
    if session:
        user_obj = session.user
        profile = backend.get_profile(str(user_obj.id))
        st.session_state["authenticated"] = True
        st.session_state["user"] = {
            "id": str(user_obj.id),
            "email": user_obj.email,
            "first_name": profile.get("first_name", "") if profile else "",
            "last_name": profile.get("last_name", "") if profile else "",
            "student_id": profile.get("student_id", "") if profile else "",
        }

# DEV MODE: bypass login screen (re-enable before production push)
_DEV_BYPASS_AUTH = False

if not _DEV_BYPASS_AUTH:
    if not st.session_state["authenticated"]:
        _render_auth_page()
        st.stop()

if _DEV_BYPASS_AUTH and not st.session_state["authenticated"]:
    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "id": "dev-user",
        "email": "dev@localhost",
        "first_name": "Dev",
        "last_name": "User",
        "student_id": "000000",
    }

_current_user = st.session_state["user"]
_user_id = _current_user["id"]

# ── Cached data fetchers ────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker):
    return get_provider().get_company_info(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_metrics(ticker, mode, _v=3):
    return get_provider().get_key_metrics(ticker, mode=mode)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price(ticker, period):
    return get_provider().get_price_history(ticker, period=period)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_financials(ticker):
    return get_provider().get_annual_financials(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_balance(ticker):
    return get_provider().get_balance_sheet(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_cashflow(ticker):
    return get_provider().get_cash_flow(ticker)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote(ticker):
    return get_provider().get_quote(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ma(ticker, window=200):
    """Return the N-day simple moving average for a ticker."""
    df = get_provider().get_price_history(ticker, period="1y")
    if df is not None and not df.empty and len(df) >= window:
        return float(df['Close'].rolling(window).mean().iloc[-1])
    return None

def fetch_200ma(ticker):
    return fetch_ma(ticker, 200)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_upcoming_earnings(from_date: str, to_date: str) -> list:
    return fetch_earnings_calendar(from_date, to_date)


# ── Chart helpers ───────────────────────────────────────────────────

CHART_CFG = {"displayModeBar": False}
BAR_COLOR = "#4A90D9"
COMPARISON_COLORS = [
    "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#3498DB", "#E91E63", "#00BCD4",
]


def _compute_cagr(start, end, years):
    """CAGR if both values positive, else simple % change, else None."""
    if start is None or end is None or years <= 0:
        return None
    if start != start or end != end:
        return None
    if start > 0 and end > 0:
        return (end / start) ** (1 / years) - 1
    if start != 0:
        return (end - start) / abs(start)
    return None


def _fmt_dollars(val):
    """Format a dollar value with appropriate suffix (T/B/M)."""
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val / 1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def _fmt_pct(val):
    """Format a decimal ratio as a percentage (e.g. 0.30 → '30.00%')."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def _fmt_ratio(val):
    """Format a ratio to 2 decimals."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _auto_scale(values_in_millions):
    """Auto-detect best unit for values already in millions.
    Returns (scaled_values, suffix)."""
    max_abs = max((abs(v) for v in values_in_millions if v is not None), default=0)
    if max_abs >= 1000:
        return [v / 1000 if v is not None else 0 for v in values_in_millions], "B"
    if max_abs >= 1:
        return list(values_in_millions), "M"
    return [v * 1000 if v is not None else 0 for v in values_in_millions], "K"


def _bar(dates, values, title, prefix="$", suffix="M", decimals=2, auto_scale=False, color=None):
    """Plotly bar chart with zoom disabled."""
    if auto_scale:
        values, suffix = _auto_scale(values)
    fmt = f",.{decimals}f"
    fig = go.Figure(data=[
        go.Bar(
            x=dates, y=values,
            marker_color=color or BAR_COLOR,
            hovertemplate=f"%{{x}}<br>{prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
    )
    return fig


def _stacked_bar(dates, bottom_vals, top_vals, bottom_name, top_name,
                 bottom_color, top_color, title, prefix="$", auto_scale=False):
    """Stacked bar chart (bottom + top)."""
    suffix = "M"
    if auto_scale:
        all_vals = [v for v in bottom_vals + top_vals if v is not None]
        max_abs = max((abs(v) for v in all_vals), default=0)
        if max_abs >= 1000:
            suffix, divisor = "B", 1000
        elif max_abs >= 1:
            suffix, divisor = "M", 1
        else:
            suffix, divisor = "K", 1 / 1000
        bottom_vals = [v / divisor if v is not None else 0 for v in bottom_vals]
        top_vals = [v / divisor if v is not None else 0 for v in top_vals]

    fmt = ",.2f"
    ht = f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>"
    fig = go.Figure(data=[
        go.Bar(x=dates, y=bottom_vals, name=bottom_name,
               marker_color=bottom_color, hovertemplate=ht),
        go.Bar(x=dates, y=top_vals, name=top_name,
               marker_color=top_color, hovertemplate=ht),
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        barmode="stack",
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _bar_with_line(dates, bar_values, line_values, title, bar_name="Revenue",
                   line_name="COGS", bar_color=None, line_color="#E74C3C",
                   prefix="$", suffix=None, auto_scale=False):
    """Bar chart with an overlaid line trace."""
    if bar_color is None:
        bar_color = BAR_COLOR
    if suffix is None:
        suffix = "M"
    if auto_scale:
        all_vals = [v for v in bar_values + line_values if v is not None]
        max_abs = max((abs(v) for v in all_vals), default=0)
        if max_abs >= 1000:
            suffix, divisor = "B", 1000
        elif max_abs >= 1:
            suffix, divisor = "M", 1
        else:
            suffix, divisor = "K", 1 / 1000
        bar_values = [v / divisor if v is not None else 0 for v in bar_values]
        line_values = [v / divisor if v is not None else 0 for v in line_values]

    fmt = ",.2f"
    ht = f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>"
    fig = go.Figure(data=[
        go.Bar(x=dates, y=bar_values, name=bar_name,
               marker_color=bar_color, hovertemplate=ht),
        go.Scatter(x=dates, y=line_values, name=line_name,
                   mode="lines+markers",
                   line=dict(color=line_color, width=2),
                   marker=dict(size=6),
                   hovertemplate=ht),
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _metric_line(dates, values, title, prefix="", suffix="%", decimals=1):
    """Plotly line chart for ratio metrics."""
    fmt = f",.{decimals}f"
    fig = go.Figure(data=[
        go.Scatter(
            x=dates, y=values,
            mode="lines+markers",
            line=dict(color=BAR_COLOR, width=2),
            hovertemplate=f"%{{x}}<br>{prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
    )
    return fig


def _grouped_bar(dates, traces, title, prefix="$", auto_scale=False, stacked=False, suffix=None):
    """Plotly grouped/stacked bar chart with multiple traces and legend toggle.
    traces = list of {"name", "values", "color"} dicts."""
    if auto_scale:
        # Find shared max across all traces to pick a common unit
        all_vals = [v for t in traces for v in t["values"] if v is not None]
        max_abs = max((abs(v) for v in all_vals), default=0)
        if max_abs >= 1000:
            suffix, divisor = "B", 1000
        elif max_abs >= 1:
            suffix, divisor = "M", 1
        else:
            suffix, divisor = "K", 1 / 1000
        for t in traces:
            t["values"] = [v / divisor if v is not None else 0 for v in t["values"]]
    else:
        suffix = suffix if suffix is not None else "M"

    fmt = ",.2f"
    bars = []
    for t in traces:
        bars.append(go.Bar(
            x=dates, y=t["values"], name=t["name"],
            marker_color=t["color"],
            hovertemplate=f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
        ))
    fig = go.Figure(data=bars)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        barmode="stack" if stacked else "group",
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _comparison_bar(tickers, values, colors, title):
    """Vertical bar chart comparing a single metric across tickers.
    values may contain None for unavailable data.
    Includes full-width average (red) and median (yellow) lines."""
    import statistics as _stats
    display_vals = [v if v is not None else 0 for v in values]
    text_labels = [f"{v:.2f}" if v is not None else "N/A" for v in values]
    bar_colors = ["#1B7A3D"] * len(tickers)

    valid = [v for v in values if v is not None]
    avg_val = sum(valid) / len(valid) if valid else None
    med_val = _stats.median(valid) if valid else None

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(
        x=tickers, y=display_vals,
        marker_color=bar_colors,
        text=text_labels,
        textposition="outside",
        hovertemplate="%{x}: %{text}<extra></extra>",
        showlegend=False,
    ))

    # Full-width average & median lines using shapes (paper-relative x)
    shapes = []
    annotations = []
    if avg_val is not None:
        shapes.append(dict(
            type="line", xref="paper", yref="y",
            x0=0, x1=1, y0=avg_val, y1=avg_val,
            line=dict(color="rgba(231,76,60,0.7)", width=2, dash="dash"),
        ))
        annotations.append(dict(
            xref="paper", yref="y", x=1, y=avg_val,
            text=f"Avg: {avg_val:.2f}", showarrow=False,
            font=dict(color="rgba(231,76,60,0.9)", size=11),
            xanchor="left", xshift=4, bgcolor="rgba(0,0,0,0.85)",
            borderpad=4,
        ))
    if med_val is not None:
        shapes.append(dict(
            type="line", xref="paper", yref="y",
            x0=0, x1=1, y0=med_val, y1=med_val,
            line=dict(color="rgba(243,156,18,0.7)", width=2, dash="dot"),
        ))
        annotations.append(dict(
            xref="paper", yref="y", x=1, y=med_val,
            text=f"Med: {med_val:.2f}", showarrow=False,
            font=dict(color="rgba(243,156,18,0.9)", size=11),
            xanchor="left", xshift=4, bgcolor="rgba(0,0,0,0.85)",
            borderpad=4,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True),
        height=300,
        margin=dict(l=60, r=60, t=35, b=40),
        shapes=shapes,
        annotations=annotations,
    )
    # Ensure text labels aren't clipped — pad y-axis above tallest bar
    all_display = [v for v in display_vals if v]
    if all_display:
        y_max = max(all_display)
        fig.update_layout(yaxis=dict(range=[0, y_max * 1.15]))
    return fig


def _stacked_grouped_bar(dates, cash_vals, debt_vals, lease_vals, title, auto_scale=False):
    """Cash as standalone bar, Debt + Capital Lease stacked together."""
    prefix = "$"
    if auto_scale:
        all_vals = [v for vs in (cash_vals, debt_vals, lease_vals) for v in vs if v is not None]
        max_abs = max((abs(v) for v in all_vals), default=0)
        if max_abs >= 1000:
            suffix, divisor = "B", 1000
        elif max_abs >= 1:
            suffix, divisor = "M", 1
        else:
            suffix, divisor = "K", 1 / 1000
        cash_vals = [v / divisor if v is not None else 0 for v in cash_vals]
        debt_vals = [v / divisor if v is not None else 0 for v in debt_vals]
        lease_vals = [v / divisor if v is not None else 0 for v in lease_vals]
    else:
        suffix = "M"

    fmt = ",.2f"
    ht = f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>"
    ht_lease = f"%{{fullData.name}}: {prefix}%{{customdata[0]:{fmt}}}{suffix}<extra></extra>"
    fig = go.Figure(data=[
        go.Bar(x=dates, y=cash_vals, name="Cash",
               marker_color="#2ECC71", offsetgroup=0, hovertemplate=ht),
        go.Bar(x=dates, y=debt_vals, name="Debt",
               marker_color="#E74C3C", offsetgroup=1, hovertemplate=ht),
        go.Bar(x=dates, y=lease_vals, name="Capital Lease",
               marker_color="#E67E22", offsetgroup=1, base=debt_vals,
               customdata=[[v] for v in lease_vals], hovertemplate=ht_lease),
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        barmode="group",
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _multi_line(dates, traces, title, prefix="", suffix="%"):
    """Plotly multi-line chart with legend toggle.
    traces = list of {"name", "values", "color"} dicts."""
    fmt = ",.2f"
    lines = []
    for t in traces:
        lines.append(go.Scatter(
            x=dates, y=t["values"], name=t["name"],
            mode="lines+markers",
            line=dict(color=t["color"], width=2),
            hovertemplate=f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
        ))
    fig = go.Figure(data=lines)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _multi_line_with_avg(dates, traces, title, prefix="", suffix="",
                        show_avg=True, avg_years=None, mode="annual"):
    """Multi-line chart with a dashed horizontal line for each trace's average.
    avg_years: if set (3, 5, or 10), only average the last N years of data."""
    import statistics
    fmt = ",.2f"
    pts_per_year = 4 if mode in ("quarterly", "ttm") else 1
    lines = []
    for t in traces:
        # Main data line
        lines.append(go.Scatter(
            x=dates, y=t["values"], name=t["name"],
            mode="lines+markers",
            line=dict(color=t["color"], width=2),
            hovertemplate=f"%{{fullData.name}}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
        ))
        # Dashed average line
        if show_avg:
            valid = [v for v in t["values"] if v is not None and v == v]  # exclude NaN
            if avg_years and len(valid) > avg_years * pts_per_year:
                valid = valid[-(avg_years * pts_per_year):]
                avg_label = f"{avg_years}Y Avg"
            else:
                avg_label = "Avg"
            if valid:
                avg = statistics.mean(valid)
                lines.append(go.Scatter(
                    x=dates, y=[avg] * len(dates),
                    name=f"{t['name']} {avg_label} ({avg:.2f})",
                    mode="lines",
                    line=dict(color=t["color"], width=1.5, dash="dash"),
                    hovertemplate=f"{t['name']} {avg_label}: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
                ))
    fig = go.Figure(data=lines)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        hovermode="x unified",
        xaxis=dict(fixedrange=True, type="category"),
        yaxis=dict(fixedrange=True, tickprefix=prefix, ticksuffix=suffix,
                   tickformat=","),
        height=280,
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _add_logo(fig):
    """Add VFC logo watermark to top-left of chart."""
    if _logo_b64:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{_logo_b64}",
                xref="paper", yref="paper",
                x=0, y=1.0,
                sizex=0.10, sizey=0.10,
                xanchor="left", yanchor="top",
                opacity=0.6,
                layer="above",
            )
        )


@st.dialog("Powered by the Viking Fund Club", width="large")
def _show_fullscreen(fig, key, series_data=None, mode="annual",
                     toggles=None, radios=None, chart_builder=None):
    if chart_builder and (toggles or radios):
        def _sync_fs_toggle(fs_key, main_key):
            st.session_state[main_key] = st.session_state[fs_key]

        _n_controls = len(toggles or []) + len(radios or [])
        _t_cols = st.columns(_n_controls)
        _ci = 0
        for t in (toggles or []):
            with _t_cols[_ci]:
                dk = f"fs_{t['key']}"
                if dk not in st.session_state:
                    st.session_state[dk] = st.session_state.get(t['key'], False)
                st.toggle(t["label"], key=dk,
                          on_change=_sync_fs_toggle, args=(dk, t['key']))
            _ci += 1
        for r in (radios or []):
            with _t_cols[_ci]:
                dk = f"fs_{r['key']}"
                if dk not in st.session_state:
                    st.session_state[dk] = st.session_state.get(r['key'], r['options'][0])
                st.radio(r["label"], r["options"], key=dk, horizontal=True,
                         on_change=_sync_fs_toggle, args=(dk, r['key']))
            _ci += 1
        # Builders read main keys, which are synced via on_change callbacks
        result = chart_builder()
        if isinstance(result, tuple):
            fig, series_data = result
        else:
            fig = result
    fig_full = go.Figure(fig)
    _add_logo(fig_full)
    fig_full.update_layout(
        height=400,
        xaxis=dict(autorange=True, fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_full, width="stretch", config=CHART_CFG, key=f"full_{key}")

    if series_data:
        # For quarterly/TTM data each point is a quarter, so multiply by 4
        pts_per_year = 4 if mode in ("quarterly", "ttm") else 1
        periods = [("1Y", 1), ("2Y", 2), ("5Y", 5), ("10Y", 10)]
        chips = []
        for s in series_data:
            vals = [v for v in s["values"] if v is not None and v == v]
            if len(vals) < 2:
                continue
            parts = []
            for label, yrs in periods:
                steps = yrs * pts_per_year
                if len(vals) > steps:
                    start = vals[-(steps + 1)]
                    end = vals[-1]
                    cagr = _compute_cagr(start, end, yrs)
                    if cagr is not None:
                        pct = cagr * 100
                        sign = "+" if pct >= 0 else ""
                        if pct >= 0:
                            bg, fg = "rgba(46,204,113,0.2)", "#27ae60"
                        else:
                            bg, fg = "rgba(231,76,60,0.2)", "#e74c3c"
                        parts.append(
                            f'{label}: <span style="display:inline-block;padding:2px 8px;'
                            f'border-radius:4px;font-weight:600;background:{bg};color:{fg};">'
                            f'{sign}{pct:.2f}%</span>'
                        )
                    else:
                        parts.append(
                            f'{label}: <span style="display:inline-block;padding:2px 8px;'
                            f'border-radius:4px;font-weight:600;background:rgba(150,150,150,0.2);'
                            f'color:#888;">N/A</span>'
                        )
                else:
                    parts.append(
                        f'{label}: <span style="display:inline-block;padding:2px 8px;'
                        f'border-radius:4px;font-weight:600;background:rgba(150,150,150,0.2);'
                        f'color:#888;">N/A</span>'
                    )
            chips.append(
                f'<div style="background:rgba(150,150,150,0.1);border:1px solid rgba(150,150,150,0.25);'
                f'border-radius:8px;padding:8px 14px;text-align:center;">'
                f'<strong>{s["name"]}</strong><br>{" &middot; ".join(parts)}</div>'
            )
        if chips:
            html = (
                '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:8px;'
                'margin-top:4px;font-size:0.95em;">'
                + "".join(chips)
                + "</div>"
            )
            st.markdown(html, unsafe_allow_html=True)


def _display_chart(fig, key, series_data=None, mode="annual",
                   toggles=None, radios=None, chart_builder=None):
    """Display a chart with an expand button that opens it in a fullscreen dialog."""
    col_chart, col_btn = st.columns([40, 1])
    with col_chart:
        # Preview: limit visible range to last 10 data points
        preview = go.Figure(fig)
        _add_logo(preview)
        n_points = len(preview.data[0].x) if preview.data else 0
        is_category = preview.layout.xaxis.type == "category"
        if n_points > 10 and is_category:
            preview.update_layout(xaxis=dict(range=[n_points - 10 - 0.5, n_points - 0.5]))
        st.plotly_chart(preview, width="stretch", config=CHART_CFG, key=f"chart_{key}")
    with col_btn:
        if st.button("⛶", key=f"expand_{key}", help="Expand chart"):
            # Clear stale fs_ keys so dialog reinitializes from current main state
            for t in (toggles or []):
                st.session_state.pop(f"fs_{t['key']}", None)
            for r in (radios or []):
                st.session_state.pop(f"fs_{r['key']}", None)
            _show_fullscreen(fig, key, series_data, mode=mode,
                             toggles=toggles, radios=radios,
                             chart_builder=chart_builder)


def _line(series, title, prefix="$", company_logo_url=None, positive=True):
    """Plotly line chart for price history with conditional shading."""
    if positive:
        line_color = "#2ECC71"
        fill_color = "rgba(46,204,113,0.15)"
    else:
        line_color = "#E74C3C"
        fill_color = "rgba(231,76,60,0.15)"
    fig = go.Figure(data=[
        go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"%{{x}}<br>{prefix}%{{y:,.2f}}<extra></extra>",
        )
    ])
    # Compute padded y-axis range so the chart isn't flattened to zero
    y_min = float(series.min())
    y_max = float(series.max())
    y_span = y_max - y_min
    padding = y_span * 0.05 if y_span > 0 else abs(y_max) * 0.02

    if company_logo_url:
        fig.add_layout_image(
            dict(
                source=company_logo_url,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                sizex=0.5, sizey=0.5,
                xanchor="center", yanchor="middle",
                opacity=0.08,
                layer="below",
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(fixedrange=True, range=[series.index.min(), series.index.max()]),
        yaxis=dict(fixedrange=True, tickprefix=prefix,
                   range=[y_min - padding, y_max + padding]),
        height=350,
        margin=dict(l=60, r=20, t=35, b=40),
    )
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────

# Sidebar logo (already loaded above)
if _logo_b64:
    st.sidebar.markdown(
        f'<div style="text-align:center; margin-bottom:0.5rem; padding:0 4px;">'
        f'<img src="data:image/png;base64,{_logo_b64}" style="width:100%; max-width:100%;">'
        f'</div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    '<div style="text-align:center; font-size:1.1rem; font-weight:700; '
    'line-height:1.3; margin-bottom:0.5rem;">The Official VFC<br>Research Dashboard</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f'<div style="text-align:center; font-size:0.9rem; opacity:0.8; margin-bottom:0.5rem;">'
    f'Welcome, {_esc(_current_user["first_name"])}</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button("Logout", width="stretch", key="logout_btn"):
    backend.sign_out()
    st.rerun()

st.sidebar.divider()

# Custom CSS for sidebar navigation buttons
st.markdown("""
<style>
/* ── Sidebar nav: lock every layer to prevent any movement ── */

/* Element containers holding buttons: fixed spacing */
section[data-testid="stSidebar"] .stElementContainer:has(.stButton) {
    margin: 0 !important;
    padding: 2px 0 !important;
    min-height: 0 !important;
}

/* The .stButton wrapper div */
section[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    padding: 0 !important;
}

/* ALL sidebar buttons — identical box model, no kind-switching */
section[data-testid="stSidebar"] .stButton button {
    height: 2.5rem !important;
    min-height: 2.5rem !important;
    max-height: 2.5rem !important;
    padding: 0 0.75rem !important;
    margin: 0 !important;
    border-width: 1px !important;
    border-style: solid !important;
    border-radius: 0.5rem !important;
    box-sizing: border-box !important;
    font-weight: 400 !important;
    font-size: inherit !important;
    line-height: 1 !important;
    box-shadow: none !important;
    outline: none !important;
    transition: none !important;
    transform: none !important;
    background: rgba(128, 128, 128, 0.1) !important;
    border-color: rgba(128, 128, 128, 0.25) !important;
    color: inherit !important;
}

/* Lock inner p tag */
section[data-testid="stSidebar"] .stButton button p {
    margin: 0 !important;
    padding: 0 !important;
}

/* Hover */
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(128, 128, 128, 0.2) !important;
    transform: none !important;
}

/* Kill focus/active animations */
section[data-testid="stSidebar"] .stButton button:focus,
section[data-testid="stSidebar"] .stButton button:active {
    box-shadow: none !important;
    outline: none !important;
    transform: none !important;
}

/* Active nav highlight — marker wrapper + sibling selector */
section[data-testid="stSidebar"] .stElementContainer:has(.nav-active-wrap) + .stElementContainer .stButton button {
    background: rgba(74, 144, 217, 0.15) !important;
    border-color: #4A90D9 !important;
}
section[data-testid="stSidebar"] .stElementContainer:has(.nav-active-wrap) + .stElementContainer .stButton button:hover {
    background: rgba(74, 144, 217, 0.25) !important;
}

/* Marker wrapper: absolutely positioned, zero layout impact */
section[data-testid="stSidebar"] .stElementContainer:has(.nav-active-wrap) {
    position: absolute !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
}

/* Section header between nav groups */
section[data-testid="stSidebar"] .stElementContainer:has(.nav-section-label) {
    margin: 0 !important;
    padding: 4px 0 2px 0 !important;
    min-height: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize nav state
st.session_state.setdefault("nav_page", "Home")
if st.session_state.get("nav_page") == "\u2B50  My Watchlist":
    st.session_state["nav_page"] = "\u2B50  My Watchlists"

def _nav_btn(label, key, nav_value, *, match_contains=False):
    """Render a sidebar nav button. All secondary; active highlighted via CSS sibling."""
    cur = st.session_state["nav_page"]
    selected = (nav_value in cur) if match_contains else (cur == nav_value)
    if selected:
        st.sidebar.markdown('<div class="nav-active-wrap"></div>', unsafe_allow_html=True)
    if st.sidebar.button(label, key=key, type="secondary", width="stretch"):
        st.session_state["nav_page"] = nav_value
        st.rerun()

_nav_btn("🏠  Home", "home_btn", "Home")
_nav_btn("\u2B50  My Watchlists", "wl_nav_btn", "\u2B50  My Watchlists",
         match_contains=True)
_nav_btn("\U0001F4DD  Research Notes", "notes_nav_btn", "\U0001F4DD  Research Notes",
         match_contains=True)

st.sidebar.markdown(
    '<div class="nav-section-label" style="font-size:0.85rem; font-weight:600; '
    'opacity:0.7;">Begin Researching Below</div>',
    unsafe_allow_html=True,
)

_nav_btn("\U0001F4C8  Company Financials", "fin_nav_btn",
         "\U0001F4C8  Company Financials", match_contains=True)
_nav_btn("\u2716  Valuation Multiples", "val_nav_btn",
         "\u2716  Valuation Multiples", match_contains=True)

page = st.session_state["nav_page"]

# ── Page: Home (Landing) ─────────────────────────────────────────

if "Home" in page:
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    _hl, _hm, _hr = st.columns([1, 2, 1])
    with _hm:
        if _logo_b64:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:1.5rem;">'
                f'<img src="data:image/png;base64,{_logo_b64}" style="width:280px;">'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<h2 style="text-align:center; margin-bottom:0.25rem;">'
            'Welcome to the Official VFC Research Dashboard</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center; opacity:0.7; margin-bottom:2rem;">'
            'Use the sidebar to navigate to Company Financials or Valuation Multiples.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    _rl, _rm, _rr = st.columns([1, 3, 1])
    with _rm:
        st.markdown(
            '<p style="opacity:0.7; font-style:italic; margin-bottom:0.5rem;">'
            'This is a reminder of the trading rules and restrictions for the challenge.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("### Trading Rules & Restrictions")
        st.markdown("**Violating these rules will result in disqualification.**")
        st.markdown("""<ol style="color: inherit;">
<li><b>No Options Trading</b> — Equities (stocks) and ETFs only (this will be automatically disabled, so it shouldn't be a problem).</li>
<li><b>No Cryptocurrency</b> — If you feel the need to have some exposure, we recommend purchasing crypto ETFs or proxies (e.g. BMNR, MSTR, FBTC).</li>
<li><b>Minimum Market Cap</b> — The minimum market cap of a stock must be <b>$5 billion</b> at the time of purchase. It may drop below this amount, but then you would technically not be allowed to purchase more shares while it is under $5 billion.</li>
<li><b>The 90% Rule</b> — Teams must have at least <b>90%</b> of their capital invested at all times. You cannot sit in cash to avoid market volatility.</li>
<li><b>Margin</b> — Margin (having increased buying power) is disabled. You must work with the allotted <b>$100,000</b>.</li>
<li><b>Short Selling</b> — Short selling is allowed.</li>
<li><b>No Trade Limit</b> — There is no limit to the number of trades you can take. <em>(Recommendation: limit frequent trading. The goal is to base investments on fundamental research.)</em></li>
<li><b>Diversification Limit</b> — The maximum a position can be is <b>20%</b> of the total portfolio. This is meant to force some diversification.</li>
</ol>""", unsafe_allow_html=True)

    st.markdown(
        '<h1 style="text-align:center; margin-top:2rem; margin-bottom:1rem; font-size:2.5rem;">'
        'Happy Researching!</h1>',
        unsafe_allow_html=True,
    )

    st.stop()

# ── Page: Valuation Multiples ──────────────────────────────────────

if "Valuation Multiples" in page:
    st.markdown(
        '<h2 style="text-align:center;margin-bottom:0.5rem;">Enter Tickers to Compare Below</h2>',
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("comp_tickers", [])
    st.session_state.setdefault("comp_group_name", None)
    _at_limit = len(st.session_state["comp_tickers"]) >= 10

    # ── Saved comparison groups ───────────────────────────────
    _comp_groups = backend.get_comparison_groups(_user_id)
    _group_options = ["— New —"] + [g["name"] for g in _comp_groups]
    _cg1, _cg2, _cg3, _cg4 = st.columns([3, 1, 1, 1])
    with _cg1:
        _sel_group = st.selectbox(
            "Saved Groups", _group_options,
            index=0, label_visibility="collapsed", key="comp_group_sel",
        )
    with _cg2:
        _load_clicked = st.button("Load", key="comp_load_btn",
                                  disabled=(_sel_group == "— New —"))
    with _cg3:
        _save_clicked = st.button("Save", key="comp_save_btn",
                                  disabled=(not st.session_state["comp_tickers"]))
    with _cg4:
        _delete_clicked = st.button("Delete", key="comp_delete_btn",
                                    disabled=(_sel_group == "— New —"))

    # Handle Load
    if _load_clicked and _sel_group != "— New —":
        _match = next((g for g in _comp_groups if g["name"] == _sel_group), None)
        if _match:
            st.session_state["comp_tickers"] = list(_match["tickers"])
            st.session_state["comp_group_name"] = _match["name"]
            st.rerun()

    # Handle Save
    if _save_clicked and st.session_state["comp_tickers"]:
        st.session_state["comp_saving"] = True

    if st.session_state.get("comp_saving"):
        _sv1, _sv2 = st.columns([3, 1])
        with _sv1:
            _save_name = st.text_input(
                "Group name", value=st.session_state.get("comp_group_name") or "",
                placeholder="Enter group name", label_visibility="collapsed",
                key="comp_save_name_input",
            )
        with _sv2:
            if st.button("Confirm Save", key="comp_confirm_save"):
                if _save_name.strip():
                    res = backend.save_comparison_group(
                        _user_id, _save_name.strip(),
                        st.session_state["comp_tickers"],
                    )
                    if res["success"]:
                        st.session_state["comp_group_name"] = _save_name.strip()
                        st.session_state["comp_saving"] = False
                        st.rerun()
                    else:
                        st.error(res["error"])
                else:
                    st.warning("Please enter a group name.")

    # Handle Delete
    if _delete_clicked and _sel_group != "— New —":
        _match = next((g for g in _comp_groups if g["name"] == _sel_group), None)
        if _match:
            backend.delete_comparison_group(_match["id"], _user_id)
            if st.session_state.get("comp_group_name") == _sel_group:
                st.session_state["comp_group_name"] = None
            st.rerun()

    col_add_input, col_add_spacer = st.columns([2, 6])
    with col_add_input:
        new_ticker = _sanitize_ticker(st.text_input(
            "Ticker", placeholder="Enter Ticker",
            label_visibility="collapsed", key="comp_ticker_input",
        ))
    if new_ticker and not _at_limit and new_ticker not in st.session_state["comp_tickers"]:
        st.session_state["comp_tickers"].append(new_ticker)
        st.rerun()

    if st.session_state["comp_tickers"]:
        cols = st.columns(min(len(st.session_state["comp_tickers"]), 10))
        for i, t in enumerate(st.session_state["comp_tickers"]):
            with cols[i % len(cols)]:
                _clr = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
                st.markdown(
                    f'<div style="background:rgba(150,150,150,0.15);border:1px solid rgba(150,150,150,0.3);'
                    f'border-radius:8px;padding:8px 10px;text-align:center;margin-bottom:4px;">'
                    f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                    f'background:{_clr};margin-right:6px;vertical-align:middle;"></span>'
                    f'<strong>{t}</strong></div>',
                    unsafe_allow_html=True,
                )
                if st.button("✕", key=f"rm_{t}", use_container_width=True):
                    st.session_state["comp_tickers"].remove(t)
                    st.rerun()

        # Fetch data for all tickers
        comp_data = {}
        with st.spinner("Fetching valuation data..."):
            for _t in st.session_state["comp_tickers"]:
                try:
                    comp_data[_t] = fetch_info(_t)
                except Exception:
                    comp_data[_t] = None

        # Define the 6 metrics to chart
        _metrics = [
            ("trailing_pe", "P/E (Trailing)"),
            ("forward_pe", "Forward P/E"),
            ("price_to_sales", "P/S"),
            ("price_to_book", "P/B"),
            ("ev_to_ebitda", "EV/EBITDA"),
            ("debt_to_equity", "Debt to Equity %"),
        ]
        _tickers = st.session_state["comp_tickers"]
        for _field, _chart_title in _metrics:
            _vals = [comp_data.get(_t, {}).get(_field) if comp_data.get(_t) else None
                     for _t in _tickers]
            fig = _comparison_bar(_tickers, _vals, COMPARISON_COLORS, _chart_title)
            _display_chart(fig, f"comp_{_field}")

        # ── Summary grid: % difference from average & median ──
        import statistics as _stats

        def _pct_color(val_str):
            """Return color-coded HTML cell for a % diff value."""
            if val_str == "N/A":
                return '<td style="text-align:center;padding:6px 10px;color:rgba(255,255,255,0.5);">N/A</td>'
            pct = float(val_str.replace("%", "").replace("+", ""))
            if pct < -5:
                clr = "#2ecc71"  # green — well below avg
            elif pct < 0:
                clr = "#82e0aa"  # light green
            elif pct < 5:
                clr = "#f5b041"  # light orange
            else:
                clr = "#e74c3c"  # red — well above avg
            return f'<td style="text-align:center;padding:6px 10px;color:{clr};font-weight:600;">{val_str}</td>'

        # Build data: {metric: {ticker: {value, pct_avg, pct_med}}}
        _grid_data = {}
        for _field, _chart_title in _metrics:
            _vals = [comp_data.get(_t, {}).get(_field) if comp_data.get(_t) else None
                     for _t in _tickers]
            _valid = [v for v in _vals if v is not None]
            _avg = sum(_valid) / len(_valid) if _valid else None
            _med = _stats.median(_valid) if _valid else None
            _grid_data[_chart_title] = {}
            for i, _t in enumerate(_tickers):
                v = _vals[i]
                pct_avg = ((v - _avg) / abs(_avg) * 100) if (v is not None and _avg and _avg != 0) else None
                pct_med = ((v - _med) / abs(_med) * 100) if (v is not None and _med and _med != 0) else None
                _grid_data[_chart_title][_t] = {
                    "value": f"{v:.2f}" if v is not None else "N/A",
                    "pct_avg": f"{pct_avg:+.1f}%" if pct_avg is not None else "N/A",
                    "pct_med": f"{pct_med:+.1f}%" if pct_med is not None else "N/A",
                }

        if _grid_data:
            _col_order = [m[1] for m in _metrics]
            _tbl_style = (
                'style="width:100%;border-collapse:collapse;margin-bottom:0.5rem;"'
            )
            _th_style = 'style="text-align:center;padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.15);font-size:0.8rem;opacity:0.7;"'
            _tk_style = 'style="padding:6px 10px;font-weight:700;white-space:nowrap;"'

            def _build_table(label, key):
                header = f"<tr><th {_th_style}>Ticker</th>"
                for m in _col_order:
                    header += f"<th {_th_style}>{_esc(m)}</th>"
                header += "</tr>"
                rows_html = ""
                for _t in _tickers:
                    rows_html += f"<tr><td {_tk_style}>{_esc(_t)}</td>"
                    for m in _col_order:
                        val = _grid_data[m][_t][key]
                        if key == "value":
                            rows_html += f'<td style="text-align:center;padding:6px 10px;">{val}</td>'
                        else:
                            rows_html += _pct_color(val)
                    rows_html += "</tr>"
                return f"<table {_tbl_style}>{header}{rows_html}</table>"

            st.markdown("---")
            st.markdown("#### Comparison Summary")
            st.markdown("**Values**")
            st.markdown(_build_table("Values", "value"), unsafe_allow_html=True)
            st.markdown("**% Difference from Average**")
            st.markdown(_build_table("% Diff from Avg", "pct_avg"), unsafe_allow_html=True)
            st.markdown("**% Difference from Median**")
            st.markdown(_build_table("% Diff from Median", "pct_med"), unsafe_allow_html=True)
    else:
        st.info("Add tickers above to compare valuation multiples.")

    st.stop()

# ── Page: My Watchlists ────────────────────────────────────────────

if "My Watchlists" in page:
    st.title("My Watchlists")

    # ── Ensure all 3 groups exist ─────────────────────────────
    _all_groups = backend.ensure_default_group(_user_id)
    _groups_err = None
    if isinstance(_all_groups, tuple):
        _all_groups, _groups_err = _all_groups

    if not _all_groups or len(_all_groups) < 3:
        st.warning(
            "Could not load watchlist groups. Please check your connection "
            "or contact an administrator."
        )
        if _groups_err:
            st.code(_groups_err, language="text")
        st.stop()

    # Clamp index
    _wl_idx = st.session_state.get("wl_active_group_idx")
    if not isinstance(_wl_idx, int) or _wl_idx >= len(_all_groups):
        st.session_state["wl_active_group_idx"] = 0

    # ── Watchlist page CSS ───────────────────────────────────
    st.markdown("""
    <style>
    /* Center radio button labels within the groups box */
    [data-testid="stRadio"] > div[role="radiogroup"] {
        justify-content: center;
    }
    /* Vertically center items in watchlist table rows */
    [data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _active_group = _all_groups[st.session_state["wl_active_group_idx"]]
    st.session_state["_wl_active_group_id"] = _active_group["id"]

    # ── Pre-fetch all watchlist data (quotes + MAs) ───────────
    watchlist = backend.fetch_watchlist_cached(
        _user_id, _active_group["id"], st.session_state["wl_cache_v"])
    _wl_data = {}
    for _item in (watchlist or []):
        _t = _item["ticker"]
        try:
            _q = fetch_quote(_t)
        except Exception:
            _q = {}
        _p = _q.get("price")
        _ma50 = fetch_ma(_t, 50)
        _ma100 = fetch_ma(_t, 100)
        _ma200 = fetch_ma(_t, 200)
        _pct = lambda p, ma: ((p - ma) / ma * 100) if p and ma and ma != 0 else None
        _wl_data[_t] = {
            "price": _p,
            "change_pct": _q.get("change_pct"),
            "ma50": _ma50, "pct50": _pct(_p, _ma50),
            "ma100": _ma100, "pct100": _pct(_p, _ma100),
            "ma200": _ma200, "pct200": _pct(_p, _ma200),
        }

    # ── Top layout: groups (left) + Dip Finder chart (right) ─
    _top_left, _top_right = st.columns([1, 1])

    with _top_left:
        with st.container(border=True):
            _group_names = ["Group 1", "Group 2", "Group 3"]
            _sel_group = st.radio(
                "Watchlist Groups",
                options=_group_names,
                index=st.session_state["wl_active_group_idx"],
                horizontal=True,
                key="wl_group_radio",
            )
            _new_idx = _group_names.index(_sel_group)
            if _new_idx != st.session_state["wl_active_group_idx"]:
                st.session_state["wl_active_group_idx"] = _new_idx
                st.rerun()


    with _top_right:
        # ── Dip Finder bar chart ──────────────────────────────
        _ma_options = [50, 100, 200]
        _dip_ma = st.radio(
            "Moving Average Period",
            options=_ma_options,
            index=_ma_options.index(st.session_state.get("wl_dip_ma", 200)),
            format_func=lambda x: f"{x}-Day",
            horizontal=True,
            key="wl_dip_ma_radio",
        )
        if _dip_ma != st.session_state.get("wl_dip_ma", 200):
            st.session_state["wl_dip_ma"] = _dip_ma
            st.rerun()
        _pct_key = f"pct{_dip_ma}"

        _chart_pairs = sorted(
            [(t, _wl_data[t][_pct_key]) for t in _wl_data
             if _wl_data[t][_pct_key] is not None],
            key=lambda x: x[1],
        )
        _chart_tickers = [p[0] for p in _chart_pairs]
        _chart_pcts = [p[1] for p in _chart_pairs]
        if _chart_tickers:
            _bar_colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in _chart_pcts]
            _dip_fig = go.Figure(go.Bar(
                x=_chart_tickers,
                y=_chart_pcts,
                marker_color=_bar_colors,
                text=[f"{p:+.1f}%" for p in _chart_pcts],
                textposition="outside",
            ))
            _dip_fig.update_layout(
                title=dict(
                    text=f"Dip Finder \u2014 {_dip_ma}-Day Moving Average",
                    x=0, xanchor="left", font=dict(size=16)),
                yaxis_title=f"% from {_dip_ma}-Day MA",
                xaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=20, t=50, b=50),
                height=420,
                yaxis=dict(zeroline=True, zerolinecolor="#888", zerolinewidth=1,
                           gridcolor="rgba(128,128,128,0.2)"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(_dip_fig, use_container_width=True, config=CHART_CFG)
        else:
            st.markdown(
                '<div style="text-align:center; color:#888; padding:4rem 1rem;">'
                'Add tickers to see the Dip Finder chart</div>',
                unsafe_allow_html=True,
            )

    # ── Upcoming Earnings ──────────────────────────────────────
    if watchlist:
        _today = datetime.now().strftime("%Y-%m-%d")
        _future = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        _all_earnings = fetch_upcoming_earnings(_today, _future)

        _wl_set = {item["ticker"].upper() for item in watchlist}
        _matched = sorted(
            [e for e in _all_earnings if e.get("symbol", "").upper() in _wl_set],
            key=lambda e: e.get("date", ""),
        )

        with st.container(border=True):
            st.markdown("**Upcoming Earnings** (next 90 days)")
            if _matched:
                _eh, _ec1, _ec2, _ec3 = st.columns([1.2, 1.2, 1, 1.2])
                _eh.markdown("**Ticker**")
                _ec1.markdown("**Date**")
                _ec2.markdown("**EPS Est.**")
                _ec3.markdown("**Rev. Est.**")

                for _e in _matched:
                    _col1, _col2, _col3, _col4 = st.columns([1.2, 1.2, 1, 1.2])
                    _col1.write(_e.get("symbol", ""))
                    try:
                        _dt = datetime.strptime(_e.get("date", ""), "%Y-%m-%d")
                        _col2.write(_dt.strftime("%b %d, %Y"))
                    except (ValueError, TypeError):
                        _col2.write(_e.get("date", ""))
                    _eps_est = _e.get("epsEstimated")
                    _col3.write(f"${_eps_est:.2f}" if _eps_est is not None else "--")
                    _rev_est = _e.get("revenueEstimated")
                    if _rev_est is not None:
                        if abs(_rev_est) >= 1e9:
                            _col4.write(f"${_rev_est / 1e9:.1f}B")
                        elif abs(_rev_est) >= 1e6:
                            _col4.write(f"${_rev_est / 1e6:.0f}M")
                        else:
                            _col4.write(f"${_rev_est:,.0f}")
                    else:
                        _col4.write("--")
            else:
                st.caption("No upcoming earnings in the next 90 days.")

    # ── Add ticker (Enter to submit) ──────────────────────────
    def _handle_add_ticker():
        raw = st.session_state.get("wl_ticker_input", "")
        ticker = _sanitize_ticker(raw)
        if ticker:
            uid = st.session_state["user"]["id"]
            gid = st.session_state.get("_wl_active_group_id")
            if uid and gid:
                res = backend.add_to_watchlist(uid, ticker, gid)
                if res["success"]:
                    backend.log_activity(uid, "add_watchlist", ticker=ticker)
                    backend.invalidate_watchlist_cache()
        st.session_state["wl_ticker_input"] = ""

    _fc1, _fc2 = st.columns([2, 6])
    with _fc1:
        st.text_input(
            "Ticker", placeholder="Enter Ticker",
            label_visibility="collapsed", key="wl_ticker_input",
            on_change=_handle_add_ticker,
        )

    st.divider()

    # ── Rich data table ───────────────────────────────────────
    _TBL_COLS = [1.4, 1.8, 1.2, 1, 1.2, 1, 1.2, 1, 0.6]
    if watchlist:
        # Header row
        (_hTkr, _hPrc, _h50, _h50p, _h100, _h100p,
         _h200, _h200p, _hRm) = st.columns(_TBL_COLS)
        _ctr = lambda t: f'<div style="text-align:center;"><b>{t}</b></div>'
        _hTkr.markdown(_ctr("Ticker"), unsafe_allow_html=True)
        _hPrc.markdown(_ctr("Price"), unsafe_allow_html=True)
        _h50.markdown(_ctr("50-MA"), unsafe_allow_html=True)
        _h50p.markdown(_ctr("% 50"), unsafe_allow_html=True)
        _h100.markdown(_ctr("100-MA"), unsafe_allow_html=True)
        _h100p.markdown(_ctr("% 100"), unsafe_allow_html=True)
        _h200.markdown(_ctr("200-MA"), unsafe_allow_html=True)
        _h200p.markdown(_ctr("% 200"), unsafe_allow_html=True)

        def _ma_cell(col, val):
            col.markdown(
                f'<div style="text-align:center;">'
                f'{"${:.2f}".format(val) if val is not None else "N/A"}'
                f'</div>', unsafe_allow_html=True)

        def _pct_cell(col, val):
            if val is not None:
                _c = "#27ae60" if val >= 0 else "#e74c3c"
                _s = "+" if val >= 0 else ""
                col.markdown(
                    f'<div style="text-align:center;">'
                    f'<span style="color:{_c}; font-weight:600;">'
                    f'{_s}{val:.2f}%</span></div>', unsafe_allow_html=True)
            else:
                col.markdown(
                    '<div style="text-align:center;">N/A</div>',
                    unsafe_allow_html=True)

        for item in watchlist:
            _t = item["ticker"]
            _d = _wl_data.get(_t, {})
            _p = _d.get("price")
            _chg_pct = _d.get("change_pct")

            (_c1, _c2, _c3, _c4, _c5, _c6,
             _c7, _c8, _c9) = st.columns(_TBL_COLS)
            # Ticker — clickable link to Company Financials
            with _c1:
                if st.button(
                    _t, key=f"wl_view_{_t}_{_active_group['id']}",
                    use_container_width=True, type="secondary",
                ):
                    st.session_state["active_ticker"] = _t
                    st.session_state["nav_page"] = "\U0001F4C8  Company Financials"
                    st.rerun()
            # Price + daily %
            if _p is not None:
                _price_str = f"${_p:.2f}"
                if _chg_pct is not None:
                    _chg_color = "#27ae60" if _chg_pct >= 0 else "#e74c3c"
                    _chg_sign = "+" if _chg_pct >= 0 else ""
                    _price_str += (
                        f' <span style="color:{_chg_color}; font-size:0.85em;">'
                        f'({_chg_sign}{_chg_pct:.2f}%)</span>')
                _c2.markdown(
                    f'<div style="text-align:center;">{_price_str}</div>',
                    unsafe_allow_html=True)
            else:
                _c2.markdown(
                    '<div style="text-align:center;">N/A</div>',
                    unsafe_allow_html=True)
            # 50-Day MA + %
            _ma_cell(_c3, _d.get("ma50"))
            _pct_cell(_c4, _d.get("pct50"))
            # 100-Day MA + %
            _ma_cell(_c5, _d.get("ma100"))
            _pct_cell(_c6, _d.get("pct100"))
            # 200-Day MA + %
            _ma_cell(_c7, _d.get("ma200"))
            _pct_cell(_c8, _d.get("pct200"))
            # Remove
            with _c9:
                if st.button("\u2715", key=f"wl_rm_{_t}_{_active_group['id']}",
                             use_container_width=True):
                    backend.remove_from_watchlist(_user_id, _t, _active_group["id"])
                    backend.log_activity(_user_id, "remove_watchlist", ticker=_t)
                    st.rerun()
    else:
        st.info("This group is empty. Add tickers above to start tracking.")

    # ── Limits caption ────────────────────────────────────────
    st.caption(
        f"{len(watchlist) if watchlist else 0}/{backend.MAX_STOCKS_PER_GROUP} "
        f"stocks in this group"
    )

    st.stop()

# ── Page: Research Notes ───────────────────────────────────────────

if "Research Notes" in page:
    st.title("Research Notes")

    rn_col_filter, rn_col_btn, rn_col_spacer = st.columns([2, 1, 5])
    with rn_col_filter:
        rn_filter = _sanitize_ticker(st.text_input(
            "Filter by ticker", placeholder="Filter by ticker (optional)",
            label_visibility="collapsed", key="rn_filter_input",
        )) or None
    with rn_col_btn:
        if st.button("New Note", key="rn_new_btn"):
            st.session_state["rn_editing"] = True
            st.session_state["rn_edit_id"] = None

    # Note editor
    st.session_state.setdefault("rn_editing", False)
    st.session_state.setdefault("rn_edit_id", None)

    if st.session_state["rn_editing"]:
        with st.expander("Note Editor", expanded=True):
            # Pre-fill if editing existing note
            edit_id = st.session_state["rn_edit_id"]
            defaults = {"ticker": rn_filter or "", "title": "", "content": "", "sentiment": "Neutral"}
            if edit_id:
                existing = backend.get_notes(_user_id)
                match = [n for n in existing if n["id"] == edit_id]
                if match:
                    n = match[0]
                    defaults["ticker"] = n.get("ticker", "")
                    defaults["title"] = n.get("title", "")
                    defaults["content"] = n.get("content", "")
                    if n.get("is_bullish") is True:
                        defaults["sentiment"] = "Bullish"
                    elif n.get("is_bullish") is False:
                        defaults["sentiment"] = "Bearish"
                    else:
                        defaults["sentiment"] = "Neutral"

            ne_ticker = st.text_input("Ticker", value=defaults["ticker"], key="ne_ticker")
            ne_title = st.text_input("Title", value=defaults["title"], key="ne_title")
            ne_content = st.text_area("Content (Markdown supported)", value=defaults["content"],
                                      height=200, key="ne_content")
            ne_sentiment = st.radio("Sentiment", ["Neutral", "Bullish", "Bearish"],
                                    horizontal=True, key="ne_sentiment",
                                    index=["Neutral", "Bullish", "Bearish"].index(defaults["sentiment"]))

            ne_c1, ne_c2, _ = st.columns([1, 1, 4])
            with ne_c1:
                if st.button("Save", type="primary", key="ne_save"):
                    if not ne_ticker or not ne_title:
                        st.error("Ticker and Title are required.")
                    else:
                        is_bullish = True if ne_sentiment == "Bullish" else (
                            False if ne_sentiment == "Bearish" else None)
                        if edit_id:
                            res = backend.update_note(
                                edit_id, _user_id,
                                ticker=ne_ticker.upper(), title=ne_title,
                                content=ne_content, is_bullish=is_bullish,
                            )
                            if res["success"]:
                                backend.log_activity(_user_id, "update_note", ticker=ne_ticker.upper())
                                st.session_state["rn_editing"] = False
                                st.session_state["rn_edit_id"] = None
                                st.rerun()
                            else:
                                st.error(res["error"])
                        else:
                            res = backend.create_note(
                                _user_id, ne_ticker, ne_title, ne_content,
                                is_bullish=is_bullish,
                            )
                            if res["success"]:
                                backend.log_activity(_user_id, "create_note", ticker=ne_ticker.upper())
                                st.session_state["rn_editing"] = False
                                st.session_state["rn_edit_id"] = None
                                st.rerun()
                            else:
                                st.error(res["error"])
            with ne_c2:
                if st.button("Cancel", key="ne_cancel"):
                    st.session_state["rn_editing"] = False
                    st.session_state["rn_edit_id"] = None
                    st.rerun()

    st.divider()

    notes = backend.fetch_notes_cached(_user_id, rn_filter, st.session_state["notes_cache_v"])

    if notes:
        for note in notes:
            # Sentiment indicator
            if note.get("is_bullish") is True:
                indicator = '<span style="color:#2ECC71; font-size:1.2em;">&#9679;</span>'
                sent_label = "Bullish"
            elif note.get("is_bullish") is False:
                indicator = '<span style="color:#E74C3C; font-size:1.2em;">&#9679;</span>'
                sent_label = "Bearish"
            else:
                indicator = '<span style="color:#888; font-size:1.2em;">&#9679;</span>'
                sent_label = "Neutral"

            nc1, nc2, nc3 = st.columns([6, 1, 1])
            with nc1:
                created = note.get("created_at", "")[:10] if note.get("created_at") else ""
                st.markdown(
                    f'{indicator} **{_esc(note["ticker"])}** — {_esc(note["title"])} '
                    f'<span style="opacity:0.5; font-size:0.85em;">({sent_label} · {created})</span>',
                    unsafe_allow_html=True,
                )
            with nc2:
                if st.button("Edit", key=f"rn_edit_{note['id']}"):
                    st.session_state["rn_editing"] = True
                    st.session_state["rn_edit_id"] = note["id"]
                    st.rerun()
            with nc3:
                if st.button("Delete", key=f"rn_del_{note['id']}"):
                    backend.delete_note(note["id"], _user_id)
                    backend.log_activity(_user_id, "delete_note", ticker=note["ticker"])
                    st.rerun()

            if note.get("content"):
                st.markdown(note["content"])
            st.markdown("---")
    else:
        st.info("No research notes yet. Click 'New Note' to create one.")

    st.stop()

# ── Page: Company Financials ────────────────────────────────────────

# ── Gate: require an active ticker ──────────────────────────────────

if "active_ticker" not in st.session_state:
    st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
    _left, _mid, _right = st.columns([1, 2, 1])
    with _mid:
        ticker_input = _sanitize_ticker(st.text_input(
            "Ticker Symbol", label_visibility="collapsed",
            placeholder="Ticker Symbol", key="ticker_gate",
        ))
        st.caption("Input a Ticker and Press Enter to Analyze")
        if ticker_input:
            st.session_state["active_ticker"] = ticker_input
            st.rerun()
    st.stop()

# Ticker search at top of main area
_col_left, _col_input, _col_right = st.columns([3, 2, 3])
with _col_input:
    ticker_input = _sanitize_ticker(st.text_input(
        "Ticker Symbol", value=st.session_state["active_ticker"],
        label_visibility="collapsed", placeholder="Ticker Symbol",
    ))
    if ticker_input and ticker_input != st.session_state["active_ticker"]:
        st.session_state["active_ticker"] = ticker_input
        st.rerun()

ticker = st.session_state["active_ticker"]

# Log ticker view (once per session per ticker)
_view_key = f"viewed_{ticker}"
if _view_key not in st.session_state:
    st.session_state[_view_key] = True
    backend.log_activity(_user_id, "view_ticker", ticker=ticker)

# ── Fetch company info (cached) ────────────────────────────────────

with st.spinner(f"Fetching data for {ticker}…"):
    try:
        info = fetch_info(ticker)
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**: {e}")
        st.stop()

# ── Company header ──────────────────────────────────────────────────

logo_url = info.get("logo_url", "")
st.markdown(
    f'<h1><img src="{_esc(logo_url)}" height="40" style="vertical-align: middle; margin-right: 10px;">'
    f'{_esc(info["name"])}  ({_esc(ticker)})</h1>',
    unsafe_allow_html=True,
)

# Price + day change + after hours change
try:
    _quote = fetch_quote(ticker)
except Exception:
    _quote = {}
_q_price = _quote.get("price")
_q_change = _quote.get("change")
_q_change_pct = _quote.get("change_pct")
_q_open = _quote.get("open")
_q_prev_close = _quote.get("previous_close")

if _q_price is not None:
    _price_col, _ = st.columns([1, 1])
    with _price_col:
        _day_html = ""
        if _q_change is not None and _q_change_pct is not None:
            _day_color = "#27ae60" if _q_change >= 0 else "#e74c3c"
            _day_sign = "+" if _q_change >= 0 else ""
            _day_html = (
                f'<span style="font-size:1.1rem; font-weight:600; color:{_day_color}; margin-left:12px;">'
                f'{_day_sign}${_q_change:.2f} ({_day_sign}{_q_change_pct:.2f}%)</span>'
            )
        _sub_html = ""
        _sub_parts = []
        # Day change: open → current price
        if _q_open is not None and _q_open != 0:
            _intra_change = _q_price - _q_open
            _intra_pct = _intra_change / _q_open * 100
            _intra_color = "#27ae60" if _intra_change >= 0 else "#e74c3c"
            _intra_sign = "+" if _intra_change >= 0 else ""
            _sub_parts.append(
                f'Day: <span style="color:{_intra_color}; font-weight:600;">'
                f'{_intra_sign}${_intra_change:.2f} ({_intra_sign}{_intra_pct:.2f}%)</span>'
            )
        # After Hours change: previous close → open
        if _q_open is not None and _q_prev_close is not None and _q_prev_close != 0:
            _on_change = _q_open - _q_prev_close
            _on_pct = _on_change / _q_prev_close * 100
            _on_color = "#27ae60" if _on_change >= 0 else "#e74c3c"
            _on_sign = "+" if _on_change >= 0 else ""
            _sub_parts.append(
                f'After Hours: <span style="color:{_on_color}; font-weight:600;">'
                f'{_on_sign}${_on_change:.2f} ({_on_sign}{_on_pct:.2f}%)</span>'
            )
        if _sub_parts:
            _sub_html = (
                f'<div style="font-size:0.85rem; opacity:0.7; margin-top:2px;">'
                f'{" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(_sub_parts)}</div>'
            )
        st.markdown(
            f'<div style="margin-top:-0.5rem; margin-bottom:0.5rem;">'
            f'<span style="font-size:1.8rem; font-weight:700;">${_q_price:.2f}</span>'
            f'{_day_html}{_sub_html}</div>',
            unsafe_allow_html=True,
        )

c1, c2, c3 = st.columns(3)
c1.metric("Sector", info["sector"])
c2.metric("Industry", info["industry"])
mc = info["market_cap"]
if mc:
    mc_str = f"${mc / 1e12:.2f}T" if mc >= 1e12 else f"${mc / 1e9:.1f}B"
else:
    mc_str = "N/A"
c3.metric("Market Cap", mc_str)

# Inline watchlist group toggle
_cf_groups = backend.get_watchlist_groups(_user_id)
if not _cf_groups:
    backend.ensure_default_group(_user_id)
    _cf_groups = backend.get_watchlist_groups(_user_id)
_cf_ticker_groups = backend.get_ticker_groups(_user_id, ticker)
_cf_group_map = {g["id"]: g["name"] for g in _cf_groups}

if _cf_ticker_groups:
    # Show remove button for each group containing this ticker
    for _gid in _cf_ticker_groups:
        _gname = _cf_group_map.get(_gid, "Unknown")
        if st.button(f"Remove from {_gname}", key=f"cf_wl_rm_{_gid}"):
            backend.remove_from_watchlist(_user_id, ticker, _gid)
            backend.log_activity(_user_id, "remove_watchlist", ticker=ticker)
            st.rerun()

# Show add-to-group picker for groups that don't contain this ticker
_cf_available = [g for g in _cf_groups if g["id"] not in _cf_ticker_groups]
if _cf_available:
    _cfa1, _cfa2 = st.columns([2, 1])
    with _cfa1:
        _cf_sel = st.selectbox(
            "Add to group", range(len(_cf_available)),
            format_func=lambda i: _cf_available[i]["name"],
            label_visibility="collapsed", key="cf_wl_group_sel",
        )
    with _cfa2:
        if st.button("Add to Watchlist", key="cf_wl_add"):
            res = backend.add_to_watchlist(
                _user_id, ticker, _cf_available[_cf_sel]["id"])
            if res["success"]:
                backend.log_activity(_user_id, "add_watchlist", ticker=ticker)
                st.rerun()
            else:
                st.error(res["error"])

# ── Price History ───────────────────────────────────────────────────

st.subheader("Price History")

PERIOD_MAP = {"YTD": "ytd", "3M": "3mo", "6M": "6mo", "1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
period_label = st.radio(
    "Price range",
    list(PERIOD_MAP.keys()),
    horizontal=True,
    key="price_period",
    label_visibility="collapsed",
)

history = fetch_price(ticker, PERIOD_MAP[period_label])
if history is not None and not history.empty:
    _start_price = history["Close"].iloc[0]
    _end_price = history["Close"].iloc[-1]
    _positive = _end_price >= _start_price
    _display_chart(
        _line(history["Close"], f"{ticker} — {period_label}", company_logo_url=logo_url, positive=_positive),
        "price",
    )
    # Price change % for selected period
    if _start_price and _start_price != 0:
        _pct_change = (_end_price - _start_price) / _start_price * 100
        _sign = "+" if _pct_change >= 0 else ""
        if _pct_change >= 0:
            _color = "#27ae60"
        else:
            _color = "#e74c3c"
        st.markdown(
            f'<div style="text-align:center; font-size:1.1rem; margin-top:-0.5rem;">'
            f'{period_label} Change: '
            f'<span style="font-weight:700; color:{_color};">{_sign}{_pct_change:.2f}%</span>'
            f' &nbsp;(${_start_price:.2f} → ${_end_price:.2f})'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No price history available.")

if info["summary"]:
    with st.expander("Business Summary"):
        st.write(info["summary"])

st.divider()
# ── Snapshot: Valuation / Dividend / Margins / Net Debt ──
snap_c1, snap_c2, snap_c3, snap_c4 = st.columns(4)

with snap_c1:
    st.markdown("<u><b>Valuation</b></u>", unsafe_allow_html=True)
    tpe = _fmt_ratio(info.get("trailing_pe"))
    fpe = _fmt_ratio(info.get("forward_pe"))
    st.markdown(f"P/E  {tpe}  |  Fwd  {fpe}")
    st.markdown(f"P/S &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt_ratio(info.get('price_to_sales'))}")
    st.markdown(f"P/B &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt_ratio(info.get('price_to_book'))}")
    st.markdown(f"EV/EBITDA &nbsp;{_fmt_ratio(info.get('ev_to_ebitda'))}")

with snap_c2:
    st.markdown("<u><b>Dividend</b></u>", unsafe_allow_html=True)
    st.markdown(f"Yield &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt_pct(info.get('dividend_yield'))}")
    st.markdown(f"Payout Ratio {_fmt_pct(info.get('payout_ratio'))}")
    ex_div = info.get("ex_dividend_date")
    ex_div_str = ex_div if isinstance(ex_div, str) else "N/A"
    st.markdown(f"Ex-Div &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ex_div_str}")

with snap_c3:
    st.markdown("<u><b>Margins</b></u>", unsafe_allow_html=True)
    st.markdown(f"Oper Margin &nbsp;{_fmt_pct(info.get('operating_margin'))}")
    st.markdown(f"Profit Margin {_fmt_pct(info.get('profit_margin'))}")
    # FCF Yield
    fcf_val = info.get("free_cashflow")
    shares_val = info.get("shares_outstanding")
    price_val = info.get("current_price")
    if fcf_val and shares_val and price_val:
        fcf_yield = (fcf_val / shares_val) / price_val * 100
        fcf_yield_str = f"{fcf_yield:.2f}%"
    else:
        fcf_yield_str = "N/A"
    st.markdown(f"FCF Yield &nbsp;&nbsp;&nbsp;{fcf_yield_str}")
    st.markdown(f"EBITDA Margin {_fmt_pct(info.get('ebitda_margin'))}")

with snap_c4:
    st.markdown("<u><b>Net Debt</b></u>", unsafe_allow_html=True)
    st.markdown(f"Cash &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt_dollars(info.get('total_cash'))}")
    st.markdown(f"Debt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{_fmt_dollars(info.get('total_debt'))}")
    t_cash = info.get("total_cash")
    t_debt = info.get("total_debt")
    if t_cash is not None and t_debt is not None:
        net_debt = t_cash - t_debt
        st.markdown(f"Net Debt &nbsp;{_fmt_dollars(net_debt)}")
    else:
        st.markdown("Net Debt &nbsp;N/A")

# ── Key Fundamental Metrics ─────────────────────────────────────────

st.subheader("Key Fundamental Metrics")

# Styled metric mode toggle
st.markdown("""
<style>
/* Metric toggle — selected (primary): flat light green + slide-up */
@keyframes slideUp {
    from { margin-top: 2.2rem; opacity: 0.7; }
    to   { margin-top: 0; opacity: 1; }
}
[data-testid="stMainBlockContainer"] button[kind="primary"] {
    background: rgba(46, 204, 113, 0.2) !important;
    border: 1px solid #2ECC71 !important;
    color: inherit !important;
    box-shadow: none !important;
    animation: slideUp 0.3s ease forwards;
}
[data-testid="stMainBlockContainer"] button[kind="primary"]:hover {
    background: rgba(46, 204, 113, 0.3) !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("metrics_mode_sel", "Annual")
_, _toggle_col, _ = st.columns([2, 3, 2])
with _toggle_col:
    _tc = st.columns(3)
    for _i, _label in enumerate(["Annual", "TTM", "Quarterly"]):
        with _tc[_i]:
            _btn_type = "primary" if st.session_state["metrics_mode_sel"] == _label else "secondary"
            if st.button(_label, key=f"mode_{_label}", width="stretch", type=_btn_type):
                st.session_state["metrics_mode_sel"] = _label
                st.rerun()
mode = st.session_state["metrics_mode_sel"]
mode_key = mode.lower()

metrics = fetch_metrics(ticker, mode_key)

if metrics is not None and not metrics.empty:
    # Chronological order (oldest → newest, left → right)
    dates = list(metrics.columns[::-1])

    def vals(row):
        if row in metrics.index:
            return metrics.loc[row][dates].tolist()
        return [0] * len(dates)

    # Income Statement
    st.markdown("**Income Statement**")
    _t1, _t2, _t3 = st.columns(3)
    with _t1:
        show_cogs = st.toggle("Show COGS", key="show_cogs")
    with _t2:
        show_basic_eps = st.toggle("Show Undiluted EPS", key="show_basic_eps")
    with _t3:
        show_opex = st.toggle("Show OpEx", key="show_opex")
    # Builder closures for charts with toggles
    def _build_revenue():
        if st.session_state.get("show_cogs"):
            fig = _bar_with_line(
                dates, vals("Revenue ($M)"), vals("COGS ($M)"),
                title="Revenue & COGS", bar_color="#D35400", line_color="#8B0000", auto_scale=True,
            )
            sd = [{"name": "Revenue", "values": vals("Revenue ($M)")},
                  {"name": "COGS", "values": vals("COGS ($M)")}]
        else:
            fig = _bar(dates, vals("Revenue ($M)"), "Revenue", auto_scale=True, color="#D35400")
            sd = [{"name": "Revenue", "values": vals("Revenue ($M)")}]
        return fig, sd

    def _build_eps():
        if st.session_state.get("show_basic_eps"):
            fig = _bar_with_line(
                dates, vals("Diluted EPS ($)"), vals("Basic EPS ($)"),
                title="EPS", bar_name="EPS", line_name="Undiluted",
                bar_color="#D4AC0D", line_color="#2ECC71", prefix="$", suffix="", auto_scale=False,
            )
            sd = [{"name": "EPS", "values": vals("Diluted EPS ($)")},
                  {"name": "Undiluted EPS", "values": vals("Basic EPS ($)")}]
        else:
            fig = _bar(dates, vals("Diluted EPS ($)"), "EPS",
                       prefix="$", suffix="", decimals=2, color="#D4AC0D")
            sd = [{"name": "EPS", "values": vals("Diluted EPS ($)")}]
        return fig, sd

    def _build_op_income():
        if st.session_state.get("show_opex"):
            fig = _bar_with_line(
                dates, vals("Operating Income ($M)"), vals("Operating Expenses ($M)"),
                title="Operating Income & OpEx", bar_name="Op. Income",
                line_name="OpEx", line_color="#E74C3C", auto_scale=True,
            )
            sd = [{"name": "Operating Income", "values": vals("Operating Income ($M)")},
                  {"name": "Operating Expenses", "values": vals("Operating Expenses ($M)")}]
        else:
            fig = _bar(dates, vals("Operating Income ($M)"), "Operating Income", auto_scale=True)
            sd = [{"name": "Operating Income", "values": vals("Operating Income ($M)")}]
        return fig, sd

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, sd = _build_revenue()
        _display_chart(fig, "revenue", series_data=sd, mode=mode_key,
                       toggles=[{"label": "Show COGS", "key": "show_cogs"}],
                       chart_builder=_build_revenue)
    with c2:
        fig, sd = _build_eps()
        _display_chart(fig, "eps", series_data=sd, mode=mode_key,
                       toggles=[{"label": "Show Undiluted EPS", "key": "show_basic_eps"}],
                       chart_builder=_build_eps)
    with c3:
        fig, sd = _build_op_income()
        _display_chart(fig, "op_income", series_data=sd, mode=mode_key,
                       toggles=[{"label": "Show OpEx", "key": "show_opex"}],
                       chart_builder=_build_op_income)

    # EBITDA & Expenses row
    def _build_ebitda():
        if st.session_state.get("show_ebitda_breakdown"):
            fig = _stacked_bar(
                dates,
                bottom_vals=vals("EBIT ($M)"),
                top_vals=vals("D&A ($M)"),
                bottom_name="EBIT",
                top_name="D&A",
                bottom_color="#00BCD4",
                top_color="#E57373",
                title="EBITDA Breakdown",
                auto_scale=True,
            )
            sd = [{"name": "EBIT", "values": vals("EBIT ($M)")},
                  {"name": "D&A", "values": vals("D&A ($M)")}]
        else:
            fig = _bar(dates, vals("EBITDA ($M)"), "EBITDA", auto_scale=True, color="#00BCD4")
            sd = [{"name": "EBITDA", "values": vals("EBITDA ($M)")}]
        return fig, sd

    _te1, _te2 = st.columns(2)
    with _te1:
        show_ebitda_breakdown = st.toggle("EBIT & D\u200BA Breakdown", key="show_ebitda_breakdown")
    c1, c2 = st.columns(2)
    with c1:
        fig, sd = _build_ebitda()
        _display_chart(fig, "ebitda", series_data=sd, mode=mode_key,
                       toggles=[{"label": "EBIT & D\u200BA Breakdown", "key": "show_ebitda_breakdown"}],
                       chart_builder=_build_ebitda)
    with c2:
        # Split SG&A into G&A + S&M when the company reports them separately
        _ga_vals = vals("G&A ($M)") if "G&A ($M)" in metrics.index else None
        _sm_vals = vals("S&M ($M)") if "S&M ($M)" in metrics.index else None
        _has_split = _ga_vals and _sm_vals and any(v != 0 for v in _ga_vals)

        if _has_split:
            _display_chart(_grouped_bar(dates, [
                {"name": "CAPEX", "values": vals("CAPEX ($M)"), "color": "#3498DB"},
                {"name": "R&D", "values": vals("R&D ($M)"), "color": "#9B59B6"},
                {"name": "G&A", "values": _ga_vals, "color": "#E67E22"},
                {"name": "S&M", "values": _sm_vals, "color": "#E74C3C"},
            ], "Expenses", auto_scale=True, stacked=True), "expenses", series_data=[
                {"name": "CAPEX", "values": vals("CAPEX ($M)")},
                {"name": "R&D", "values": vals("R&D ($M)")},
                {"name": "G&A", "values": _ga_vals},
                {"name": "S&M", "values": _sm_vals},
            ], mode=mode_key)
        else:
            _display_chart(_grouped_bar(dates, [
                {"name": "CAPEX", "values": vals("CAPEX ($M)"), "color": "#3498DB"},
                {"name": "R&D", "values": vals("R&D ($M)"), "color": "#9B59B6"},
                {"name": "SG&A", "values": vals("SG&A ($M)"), "color": "#E67E22"},
            ], "Expenses", auto_scale=True, stacked=True), "expenses", series_data=[
                {"name": "CAPEX", "values": vals("CAPEX ($M)")},
                {"name": "R&D", "values": vals("R&D ($M)")},
                {"name": "SG&A", "values": vals("SG&A ($M)")},
            ], mode=mode_key)

    # Cash Flow
    st.markdown("**Cash Flow**")
    shares_vals = vals("Shares Outstanding (M)")
    fcf_vals = vals("FCF ($M)")
    sbc_vals = vals("SBC ($M)")

    def _build_ocf():
        if st.session_state.get("cf_per_share"):
            ocf_ps = [o / s if s else 0 for o, s in zip(vals("OCF ($M)"), shares_vals)]
            fig = _bar(dates, ocf_ps, "OCF / Share", prefix="$", suffix="", decimals=2)
            sd = [{"name": "OCF/Share", "values": ocf_ps}]
        else:
            fig = _bar(dates, vals("OCF ($M)"), "Operating Cash Flow", auto_scale=True)
            sd = [{"name": "OCF", "values": vals("OCF ($M)")}]
        return fig, sd

    def _build_fcf():
        ps = st.session_state.get("cf_per_share")
        adj = st.session_state.get("cf_adjust_sbc")
        if ps:
            fcf_ps = [f / s if s else 0 for f, s in zip(fcf_vals, shares_vals)]
            sbc_ps = [b / s if s else 0 for b, s in zip(sbc_vals, shares_vals)]
            if adj:
                fcf_adj_ps = [f - b for f, b in zip(fcf_ps, sbc_ps)]
                fig = _bar(dates, fcf_adj_ps, "FCF Adj. / Share", prefix="$", suffix="", decimals=2, color="#145A32")
                sd = [{"name": "FCF Adj./Share", "values": fcf_adj_ps}]
            else:
                fig = _grouped_bar(dates, [
                    {"name": "FCF/Share", "values": fcf_ps, "color": "#1E8449"},
                    {"name": "SBC/Share", "values": sbc_ps, "color": "#E74C3C"},
                ], "FCF & SBC / Share", suffix="")
                sd = [{"name": "FCF/Share", "values": fcf_ps},
                      {"name": "SBC/Share", "values": sbc_ps}]
        else:
            if adj:
                fcf_adj = [f - b for f, b in zip(fcf_vals, sbc_vals)]
                fig = _bar(dates, fcf_adj, "FCF Adj.", auto_scale=True, color="#145A32")
                sd = [{"name": "FCF Adj.", "values": fcf_adj}]
            else:
                fig = _grouped_bar(dates, [
                    {"name": "FCF", "values": fcf_vals, "color": "#1E8449"},
                    {"name": "SBC", "values": sbc_vals, "color": "#E74C3C"},
                ], "FCF & SBC", auto_scale=True)
                sd = [{"name": "FCF", "values": fcf_vals},
                      {"name": "SBC", "values": sbc_vals}]
        return fig, sd

    _cf_t1, _cf_t2 = st.columns(2)
    with _cf_t1:
        per_share = st.toggle("Per Share", key="cf_per_share")
    with _cf_t2:
        adjust_sbc = st.toggle("Adjust for SBC", key="cf_adjust_sbc")
    c1, c2 = st.columns(2)
    with c1:
        fig, sd = _build_ocf()
        _display_chart(fig, "ocf", series_data=sd, mode=mode_key,
                       toggles=[{"label": "Per Share", "key": "cf_per_share"}],
                       chart_builder=_build_ocf)
    with c2:
        fig, sd = _build_fcf()
        _display_chart(fig, "fcf", series_data=sd, mode=mode_key,
                       toggles=[{"label": "Per Share", "key": "cf_per_share"},
                                {"label": "Adjust for SBC", "key": "cf_adjust_sbc"}],
                       chart_builder=_build_fcf)

    # Balance Sheet
    st.markdown("**Balance Sheet**")
    _display_chart(_stacked_grouped_bar(
        dates,
        cash_vals=vals("Total Cash ($M)"),
        debt_vals=vals("Long Term Debt ($M)"),
        lease_vals=vals("Capital Lease ($M)"),
        title="Cash & Debt",
        auto_scale=True,
    ), "cash_debt", mode=mode_key)

    # Efficiency Ratios — combined multi-line charts
    st.markdown("**Efficiency & Performance Ratios**")
    c1, c2 = st.columns(2)
    with c1:
        _display_chart(_multi_line(dates, [
            {"name": "ROIC", "values": vals("ROIC (%)"), "color": "#3498DB"},
            {"name": "ROE", "values": vals("ROE (%)"), "color": "#E67E22"},
            {"name": "ROA", "values": vals("ROA (%)"), "color": "#9B59B6"},
        ], "ROIC / ROE / ROA"), "ratios", mode=mode_key)
    with c2:
        _display_chart(_multi_line(dates, [
            {"name": "Gross", "values": vals("Gross Margin (%)"), "color": "#2ECC71"},
            {"name": "Operating", "values": vals("Operating Margin (%)"), "color": "#3498DB"},
            {"name": "Net", "values": vals("Net Margin (%)"), "color": "#E67E22"},
            {"name": "EBITDA", "values": vals("EBITDA Margin (%)"), "color": "#9B59B6"},
        ], "Profitability Margins"), "margins", mode=mode_key)
    _AVG_PERIOD_MAP = {"All": None, "3Y": 3, "5Y": 5, "10Y": 10}

    def _build_liquidity():
        show = st.session_state.get("liq_avg", False)
        period = st.session_state.get("liq_avg_period", "All")
        avg_years = _AVG_PERIOD_MAP.get(period)
        fig = _multi_line_with_avg(dates, [
            {"name": "Current", "values": vals("Current Ratio"), "color": "#2ECC71"},
            {"name": "Cash", "values": vals("Cash Ratio"), "color": "#1ABC9C"},
        ], "Liquidity Ratios", suffix="x", show_avg=show,
           avg_years=avg_years, mode=mode_key)
        return fig

    def _build_solvency():
        show = st.session_state.get("solv_avg", False)
        period = st.session_state.get("solv_avg_period", "All")
        avg_years = _AVG_PERIOD_MAP.get(period)
        fig = _multi_line_with_avg(dates, [
            {"name": "D/A", "values": vals("D/A (%)"), "color": "#E74C3C"},
            {"name": "D/E", "values": vals("D/E (%)"), "color": "#C0392B"},
        ], "Solvency Ratios", suffix="%", show_avg=show,
           avg_years=avg_years, mode=mode_key)
        return fig

    c1, c2 = st.columns(2)
    with c1:
        show_liq_avg = st.toggle("Show Average", key="liq_avg")
        fig = _build_liquidity()
        _display_chart(fig, "liquidity", mode=mode_key,
                       toggles=[{"label": "Show Average", "key": "liq_avg"}],
                       radios=[{"label": "Avg Period", "key": "liq_avg_period",
                                "options": ["All", "3Y", "5Y", "10Y"]}],
                       chart_builder=_build_liquidity)
    with c2:
        show_solv_avg = st.toggle("Show Average", key="solv_avg")
        fig = _build_solvency()
        _display_chart(fig, "solvency", mode=mode_key,
                       toggles=[{"label": "Show Average", "key": "solv_avg"}],
                       radios=[{"label": "Avg Period", "key": "solv_avg_period",
                                "options": ["All", "3Y", "5Y", "10Y"]}],
                       chart_builder=_build_solvency)

    # Dilution / Buybacks
    st.markdown("**Dilution / Buybacks**")
    c1, c2 = st.columns(2)
    with c1:
        _display_chart(_bar(dates, vals("Shares Outstanding (M)"),
                             "Shares Outstanding", prefix="", auto_scale=True, color="#95A5A6"), "shares",
                       series_data=[{"name": "Shares", "values": vals("Shares Outstanding (M)")}], mode=mode_key)
    with c2:
        _display_chart(_grouped_bar(dates, [
            {"name": "Dividends", "values": vals("Dividends Paid ($M)"), "color": "#D4A574"},
            {"name": "Buybacks", "values": vals("Buybacks ($M)"), "color": "#B39DDB"},
        ], "Capital Returned", auto_scale=True), "cap_returned", series_data=[
            {"name": "Dividends", "values": vals("Dividends Paid ($M)")},
            {"name": "Buybacks", "values": vals("Buybacks ($M)")},
        ], mode=mode_key)

    st.markdown(
        '<p style="font-size:0.85rem; opacity:0.6; margin-top:0.5rem;">'
        'All dollar values in millions ($M). EPS in $/share. Ratios in %.</p>',
        unsafe_allow_html=True,
    )
else:
    st.info("No metric data available.")

# ── Full Financial Statements ───────────────────────────────────────

st.divider()
st.subheader("Full Financial Statements")
st.markdown(
    '<p style="font-size:0.85rem; opacity:0.6; margin-top:0.5rem;">'
    'Values in millions ($M). EPS in $/share. Tax Rate in %.</p>',
    unsafe_allow_html=True,
)

with st.expander("Income Statement"):
    fin = fetch_financials(ticker)
    if fin is not None and not fin.empty:
        st.dataframe(_format_statement_display(fin), width="stretch")
    else:
        st.info("No income statement data available.")

with st.expander("Balance Sheet"):
    bal = fetch_balance(ticker)
    if bal is not None and not bal.empty:
        st.dataframe(_format_statement_display(bal), width="stretch")
    else:
        st.info("No balance sheet data available.")

with st.expander("Cash Flow Statement"):
    cf = fetch_cashflow(ticker)
    if cf is not None and not cf.empty:
        st.dataframe(_format_statement_display(cf), width="stretch")
    else:
        st.info("No cash flow data available.")
