"""Viking Fund Dashboard — Fundamental Analysis"""

import streamlit as st
import plotly.graph_objects as go
import base64
from pathlib import Path
from datetime import datetime
from data_manager import get_provider, _format_statement_display
import backend

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
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Sign In", use_container_width=True, type="primary"):
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
            if st.button("Create Account", use_container_width=True, type="primary"):
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

if not st.session_state["authenticated"]:
    _render_auth_page()
    st.stop()

_current_user = st.session_state["user"]
_user_id = _current_user["id"]

# ── Cached data fetchers ────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker):
    return get_provider().get_company_info(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_metrics(ticker, mode, _v=2):
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


# ── Chart helpers ───────────────────────────────────────────────────

CHART_CFG = {"displayModeBar": False}
BAR_COLOR = "#4A90D9"


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


def _bar(dates, values, title, prefix="$", suffix="M", decimals=1, auto_scale=False):
    """Plotly bar chart with zoom disabled."""
    if auto_scale:
        values, suffix = _auto_scale(values)
    fmt = f",.{decimals}f"
    fig = go.Figure(data=[
        go.Bar(
            x=dates, y=values,
            marker_color=BAR_COLOR,
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

    fmt = ",.1f"
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
                   prefix="$", auto_scale=False):
    """Bar chart with an overlaid line trace."""
    if bar_color is None:
        bar_color = BAR_COLOR
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

    fmt = ",.1f"
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


def _grouped_bar(dates, traces, title, prefix="$", auto_scale=False, stacked=False):
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
        suffix = "M"

    fmt = ",.1f"
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

    fmt = ",.1f"
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
    fmt = ",.1f"
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


def _multi_line_with_avg(dates, traces, title, prefix="", suffix=""):
    """Multi-line chart with a dashed horizontal line for each trace's average."""
    import statistics
    fmt = ",.2f"
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
        valid = [v for v in t["values"] if v is not None and v == v]  # exclude NaN
        if valid:
            avg = statistics.mean(valid)
            lines.append(go.Scatter(
                x=dates, y=[avg] * len(dates),
                name=f"{t['name']} Avg ({avg:.2f})",
                mode="lines",
                line=dict(color=t["color"], width=1.5, dash="dash"),
                hovertemplate=f"{t['name']} Avg: {prefix}%{{y:{fmt}}}{suffix}<extra></extra>",
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


@st.dialog("Chart", width="large")
def _show_fullscreen(fig, key, series_data=None):
    fig_full = go.Figure(fig)
    fig_full.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_full, use_container_width=True, config=CHART_CFG, key=f"full_{key}")

    if series_data:
        periods = [("1Y", 1), ("2Y", 2)]
        chips = []
        for s in series_data:
            vals = [v for v in s["values"] if v is not None and v == v]
            if len(vals) < 2:
                continue
            parts = []
            for label, yrs in periods:
                if len(vals) > yrs:
                    start = vals[-(yrs + 1)]
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
            chips.append(f'<strong>{s["name"]}</strong> {" &middot; ".join(parts)}')
        if chips:
            html = (
                '<div style="text-align:center;margin-top:4px;font-size:0.95em;">'
                + " &nbsp;&nbsp; ".join(chips)
                + "</div>"
            )
            st.markdown(html, unsafe_allow_html=True)


def _display_chart(fig, key, series_data=None):
    """Display a chart with an expand button that opens it in a fullscreen dialog."""
    col_chart, col_btn = st.columns([40, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CFG, key=f"chart_{key}")
    with col_btn:
        if st.button("⛶", key=f"expand_{key}", help="Expand chart"):
            _show_fullscreen(fig, key, series_data)


def _line(series, title, prefix="$"):
    """Plotly line chart for price history."""
    fig = go.Figure(data=[
        go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            line=dict(color=BAR_COLOR, width=2),
            hovertemplate=f"%{{x}}<br>{prefix}%{{y:,.2f}}<extra></extra>",
        )
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(fixedrange=True, range=[series.index.min(), series.index.max()]),
        yaxis=dict(fixedrange=True, tickprefix=prefix),
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
    f'Welcome, {_current_user["first_name"]}</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button("Logout", use_container_width=True, key="logout_btn"):
    backend.sign_out()
    st.rerun()

st.sidebar.divider()

# Custom CSS for sidebar radio → selectable boxes with icons
st.markdown("""
<style>
/* Turn sidebar radio options into selectable boxes */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.5rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(128, 128, 128, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 0.5rem;
    padding: 0.6rem 0.75rem;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    display: flex !important;
    align-items: center;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(128, 128, 128, 0.2);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(74, 144, 217, 0.15);
    border-color: #4A90D9;
}
/* Hide the default radio dot */
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}
/* Style the Home button to match the radio boxes */
section[data-testid="stSidebar"] button[key="home_btn"],
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background: rgba(128, 128, 128, 0.1) !important;
    border: 1px solid rgba(128, 128, 128, 0.25) !important;
    border-radius: 0.5rem !important;
    padding: 0.6rem 0.75rem !important;
    color: inherit !important;
    font-weight: 400 !important;
    transition: background 0.15s, border-color 0.15s;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: rgba(128, 128, 128, 0.2) !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(74, 144, 217, 0.15) !important;
    border: 1px solid #4A90D9 !important;
    border-radius: 0.5rem !important;
    padding: 0.6rem 0.75rem !important;
    color: inherit !important;
    font-weight: 400 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize nav state
st.session_state.setdefault("nav_page", "Home")

# Home button above the "Begin Researching" label
_home_selected = st.session_state["nav_page"] == "Home"
if st.sidebar.button(
    "🏠  Home", key="home_btn", use_container_width=True,
    type="primary" if _home_selected else "secondary",
):
    st.session_state["nav_page"] = "Home"
    st.rerun()

st.sidebar.markdown(
    '<div style="font-size:0.85rem; font-weight:600; margin-bottom:0.4rem; '
    'opacity:0.7;">Begin Researching Below</div>',
    unsafe_allow_html=True,
)

_nav_options = [
    "\U0001F4C8  Company Financials",
    "\u2716  Valuation Multiples",
    "\u2B50  My Watchlist",
    "\U0001F4DD  Research Notes",
]

# If currently on Home, no radio option should appear selected — use None index
_default_idx = None if st.session_state["nav_page"] == "Home" else 0
for _i, _opt in enumerate(_nav_options):
    if st.session_state["nav_page"] in _opt:
        _default_idx = _i
        break

def _on_nav_change():
    st.session_state["nav_page"] = st.session_state["_nav_radio"]

page = st.sidebar.radio(
    "Navigation",
    _nav_options,
    index=_default_idx,
    label_visibility="collapsed",
    key="_nav_radio",
    on_change=_on_nav_change,
)

# If nav_page says Home, override radio selection
if st.session_state["nav_page"] == "Home":
    page = "Home"

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
        st.markdown("""
1. **No Options Trading** — Equities (stocks) and ETFs only (this will be automatically disabled, so it shouldn't be a problem).

2. **No Cryptocurrency** — If you feel the need to have some exposure, we recommend purchasing crypto ETFs or proxies (e.g. BMNR, MSTR, FBTC).

3. **Minimum Market Cap** — The minimum market cap of a stock must be **$5 billion** at the time of purchase. It may drop below this amount, but then you would technically not be allowed to purchase more shares while it is under $5 billion.

4. **The 90% Rule** — Teams must have at least **90%** of their capital invested at all times. You cannot sit in cash to avoid market volatility.

5. **Margin** — Margin (having increased buying power) is disabled. You must work with the allotted **$100,000**.

6. **Short Selling** — Short selling is allowed.

7. **No Trade Limit** — There is no limit to the number of trades you can take. *(Recommendation: limit frequent trading. The goal is to base investments on fundamental research.)*

8. **Diversification Limit** — The maximum a position can be is **20%** of the total portfolio. This is meant to force some diversification.
""")

    st.markdown(
        '<h1 style="text-align:center; margin-top:2rem; margin-bottom:1rem; font-size:2.5rem;">'
        'Happy Researching!</h1>',
        unsafe_allow_html=True,
    )

    st.stop()

# ── Page: Valuation Multiples ──────────────────────────────────────

if "Valuation Multiples" in page:
    st.title("Valuation Multiples")
    st.session_state.setdefault("comp_tickers", [])

    col_add_input, col_add_btn, col_add_spacer = st.columns([2, 1, 5])
    with col_add_input:
        new_ticker = st.text_input(
            "Ticker", placeholder="Enter ticker symbol",
            label_visibility="collapsed", key="comp_ticker_input",
        ).upper().strip()
    with col_add_btn:
        if st.button("Add") and new_ticker and new_ticker not in st.session_state["comp_tickers"]:
            st.session_state["comp_tickers"].append(new_ticker)
            st.rerun()

    if st.session_state["comp_tickers"]:
        cols = st.columns(min(len(st.session_state["comp_tickers"]), 8))
        for i, t in enumerate(st.session_state["comp_tickers"]):
            with cols[i % len(cols)]:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{t}**")
                if c2.button("✕", key=f"rm_{t}"):
                    st.session_state["comp_tickers"].remove(t)
                    st.rerun()
    else:
        st.info("Add tickers above to compare valuation multiples.")

    st.stop()

# ── Page: My Watchlist ─────────────────────────────────────────────

if "My Watchlist" in page:
    st.title("My Watchlist")

    wl_col_input, wl_col_btn, wl_col_spacer = st.columns([2, 1, 5])
    with wl_col_input:
        wl_ticker = st.text_input(
            "Ticker", placeholder="Enter ticker symbol",
            label_visibility="collapsed", key="wl_ticker_input",
        ).upper().strip()
    with wl_col_btn:
        if st.button("Add", key="wl_add_btn") and wl_ticker:
            res = backend.add_to_watchlist(_user_id, wl_ticker)
            if res["success"]:
                backend.log_activity(_user_id, "add_watchlist", ticker=wl_ticker)
                st.rerun()
            else:
                st.error(res["error"])

    st.divider()

    watchlist = backend.fetch_watchlist_cached(_user_id, st.session_state["wl_cache_v"])

    if watchlist:
        for item in watchlist:
            _wc1, _wc2, _wc3, _wc4 = st.columns([2, 4, 1, 1])
            with _wc1:
                st.markdown(f"**{item['ticker']}**")
            with _wc2:
                added = item.get("added_at", "")[:10] if item.get("added_at") else ""
                note_text = f" — {item['notes']}" if item.get("notes") else ""
                st.caption(f"Added {added}{note_text}")
            with _wc3:
                if st.button("View", key=f"wl_view_{item['ticker']}"):
                    st.session_state["active_ticker"] = item["ticker"]
                    st.session_state["nav_page"] = "\U0001F4C8  Company Financials"
                    st.rerun()
            with _wc4:
                if st.button("Remove", key=f"wl_rm_{item['ticker']}"):
                    backend.remove_from_watchlist(_user_id, item["ticker"])
                    backend.log_activity(_user_id, "remove_watchlist", ticker=item["ticker"])
                    st.rerun()
    else:
        st.info("Your watchlist is empty. Add tickers above to start tracking.")

    st.stop()

# ── Page: Research Notes ───────────────────────────────────────────

if "Research Notes" in page:
    st.title("Research Notes")

    rn_col_filter, rn_col_btn, rn_col_spacer = st.columns([2, 1, 5])
    with rn_col_filter:
        rn_filter = st.text_input(
            "Filter by ticker", placeholder="Filter by ticker (optional)",
            label_visibility="collapsed", key="rn_filter_input",
        ).upper().strip() or None
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
                    f'{indicator} **{note["ticker"]}** — {note["title"]} '
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
        ticker_input = st.text_input(
            "Ticker Symbol", label_visibility="collapsed",
            placeholder="Ticker Symbol", key="ticker_gate",
        ).upper().strip()
        st.caption("Input a Ticker and Press Enter to Analyze")
        if ticker_input:
            st.session_state["active_ticker"] = ticker_input
            st.rerun()
    st.stop()

# Ticker search at top of main area
_col_left, _col_input, _col_right = st.columns([3, 2, 3])
with _col_input:
    ticker_input = st.text_input(
        "Ticker Symbol", value=st.session_state["active_ticker"],
        label_visibility="collapsed", placeholder="Ticker Symbol",
    ).upper().strip()
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
    f'<h1><img src="{logo_url}" height="40" style="vertical-align: middle; margin-right: 10px;">'
    f'{info["name"]}  ({ticker})</h1>',
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

# Inline watchlist toggle
_in_wl = backend.is_in_watchlist(_user_id, ticker)
_wl_label = "Remove from Watchlist" if _in_wl else "Add to Watchlist"
if st.button(_wl_label, key="cf_wl_toggle"):
    if _in_wl:
        backend.remove_from_watchlist(_user_id, ticker)
        backend.log_activity(_user_id, "remove_watchlist", ticker=ticker)
    else:
        backend.add_to_watchlist(_user_id, ticker)
        backend.log_activity(_user_id, "add_watchlist", ticker=ticker)
    st.rerun()

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
    if ex_div is not None:
        try:
            ex_div_str = datetime.fromtimestamp(ex_div).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            ex_div_str = "N/A"
    else:
        ex_div_str = "N/A"
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

if info["summary"]:
    with st.expander("Business Summary"):
        st.write(info["summary"])

# ── Key Fundamental Metrics ─────────────────────────────────────────

st.subheader("Key Fundamental Metrics")

mode = st.radio(
    "Metric time frame",
    ["Annual", "TTM", "Quarterly"],
    horizontal=True,
    key="metrics_mode",
    label_visibility="collapsed",
)
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
    with _t3:
        show_ebitda_breakdown = st.toggle("EBIT & D\u200BA Breakdown", key="show_ebitda_breakdown")
    c1, c2, c3 = st.columns(3)
    with c1:
        if show_cogs:
            _display_chart(_bar_with_line(
                dates, vals("Revenue ($M)"), vals("COGS ($M)"),
                title="Revenue & COGS", auto_scale=True,
            ), "revenue", series_data=[
                {"name": "Revenue", "values": vals("Revenue ($M)")},
                {"name": "COGS", "values": vals("COGS ($M)")},
            ])
        else:
            _display_chart(_bar(dates, vals("Revenue ($M)"), "Revenue", auto_scale=True), "revenue",
                           series_data=[{"name": "Revenue", "values": vals("Revenue ($M)")}])
    with c2:
        _display_chart(_bar(dates, vals("Diluted EPS ($)"), "Diluted EPS",
                             prefix="$", suffix="", decimals=2), "eps",
                       series_data=[{"name": "EPS", "values": vals("Diluted EPS ($)")}])
    with c3:
        if show_ebitda_breakdown:
            _display_chart(_stacked_bar(
                dates,
                bottom_vals=vals("EBIT ($M)"),
                top_vals=vals("D&A ($M)"),
                bottom_name="EBIT",
                top_name="D&A",
                bottom_color="#4A90D9",
                top_color="#F39C12",
                title="EBITDA Breakdown",
                auto_scale=True,
            ), "ebitda", series_data=[
                {"name": "EBIT", "values": vals("EBIT ($M)")},
                {"name": "D&A", "values": vals("D&A ($M)")},
            ])
        else:
            _display_chart(_bar(dates, vals("EBITDA ($M)"), "EBITDA", auto_scale=True), "ebitda",
                           series_data=[{"name": "EBITDA", "values": vals("EBITDA ($M)")}])
    c1, c2, c3 = st.columns(3)
    with c1:
        _display_chart(_grouped_bar(dates, [
            {"name": "CAPEX", "values": vals("CAPEX ($M)"), "color": "#3498DB"},
            {"name": "R&D", "values": vals("R&D ($M)"), "color": "#9B59B6"},
            {"name": "SG&A", "values": vals("SG&A ($M)"), "color": "#E67E22"},
        ], "Expenses", auto_scale=True, stacked=True), "expenses", series_data=[
            {"name": "CAPEX", "values": vals("CAPEX ($M)")},
            {"name": "R&D", "values": vals("R&D ($M)")},
            {"name": "SG&A", "values": vals("SG&A ($M)")},
        ])

    # Cash Flow
    st.markdown("**Cash Flow**")
    per_share = st.toggle("Per Share", key="cf_per_share")
    c1, c2 = st.columns(2)
    if per_share:
        shares_vals = vals("Shares Outstanding (M)")
        ocf_ps = [o / s if s else 0 for o, s in zip(vals("OCF ($M)"), shares_vals)]
        fcf_ps = [f / s if s else 0 for f, s in zip(vals("FCF ($M)"), shares_vals)]
        with c1:
            _display_chart(_bar(dates, ocf_ps, "OCF / Share", prefix="$", suffix="", decimals=2), "ocf",
                           series_data=[{"name": "OCF/Share", "values": ocf_ps}])
        with c2:
            _display_chart(_bar(dates, fcf_ps, "FCF / Share", prefix="$", suffix="", decimals=2), "fcf",
                           series_data=[{"name": "FCF/Share", "values": fcf_ps}])
    else:
        with c1:
            _display_chart(_bar(dates, vals("OCF ($M)"), "Operating Cash Flow", auto_scale=True), "ocf",
                           series_data=[{"name": "OCF", "values": vals("OCF ($M)")}])
        with c2:
            _display_chart(_grouped_bar(dates, [
                {"name": "FCF", "values": vals("FCF ($M)"), "color": "#4A90D9"},
                {"name": "SBC", "values": vals("SBC ($M)"), "color": "#E67E22"},
            ], "FCF & SBC", auto_scale=True), "fcf", series_data=[
                {"name": "FCF", "values": vals("FCF ($M)")},
                {"name": "SBC", "values": vals("SBC ($M)")},
            ])

    # Balance Sheet
    st.markdown("**Balance Sheet**")
    _display_chart(_stacked_grouped_bar(
        dates,
        cash_vals=vals("Total Cash ($M)"),
        debt_vals=vals("Long Term Debt ($M)"),
        lease_vals=vals("Capital Lease ($M)"),
        title="Cash & Debt",
        auto_scale=True,
    ), "cash_debt", series_data=[
        {"name": "Cash", "values": vals("Total Cash ($M)")},
        {"name": "Debt", "values": vals("Long Term Debt ($M)")},
        {"name": "Capital Lease", "values": vals("Capital Lease ($M)")},
    ])

    # Efficiency Ratios — combined multi-line charts
    st.markdown("**Efficiency & Performance Ratios**")
    c1, c2 = st.columns(2)
    with c1:
        _display_chart(_multi_line(dates, [
            {"name": "ROIC", "values": vals("ROIC (%)"), "color": "#3498DB"},
            {"name": "ROE", "values": vals("ROE (%)"), "color": "#E67E22"},
            {"name": "ROA", "values": vals("ROA (%)"), "color": "#9B59B6"},
        ], "ROIC / ROE / ROA"), "ratios", series_data=[
            {"name": "ROIC", "values": vals("ROIC (%)")},
            {"name": "ROE", "values": vals("ROE (%)")},
            {"name": "ROA", "values": vals("ROA (%)")},
        ])
    c1, c2 = st.columns(2)
    with c1:
        _display_chart(_multi_line_with_avg(dates, [
            {"name": "Current", "values": vals("Current Ratio"), "color": "#2ECC71"},
            {"name": "Cash", "values": vals("Cash Ratio"), "color": "#1ABC9C"},
        ], "Liquidity Ratios", suffix="x"), "liquidity", series_data=[
            {"name": "Current Ratio", "values": vals("Current Ratio")},
            {"name": "Cash Ratio", "values": vals("Cash Ratio")},
        ])
    with c2:
        _display_chart(_multi_line_with_avg(dates, [
            {"name": "D/A", "values": vals("D/A (%)"), "color": "#E74C3C"},
            {"name": "D/E", "values": vals("D/E (%)"), "color": "#C0392B"},
        ], "Solvency Ratios", suffix="%"), "solvency", series_data=[
            {"name": "D/A", "values": vals("D/A (%)")},
            {"name": "D/E", "values": vals("D/E (%)")},
        ])

    # Dilution / Buybacks
    st.markdown("**Dilution / Buybacks**")
    c1, c2 = st.columns(2)
    with c1:
        _display_chart(_bar(dates, vals("Shares Outstanding (M)"),
                             "Shares Outstanding", prefix="", auto_scale=True), "shares",
                       series_data=[{"name": "Shares", "values": vals("Shares Outstanding (M)")}])
    with c2:
        _display_chart(_grouped_bar(dates, [
            {"name": "Dividends", "values": vals("Dividends Paid ($M)"), "color": "#D4A574"},
            {"name": "Buybacks", "values": vals("Buybacks ($M)"), "color": "#B39DDB"},
        ], "Capital Returned", auto_scale=True), "cap_returned", series_data=[
            {"name": "Dividends", "values": vals("Dividends Paid ($M)")},
            {"name": "Buybacks", "values": vals("Buybacks ($M)")},
        ])

    st.caption("All dollar values in millions ($M). EPS in $/share. Ratios in %.")
else:
    st.info("No metric data available.")

# ── Price History ───────────────────────────────────────────────────

st.subheader("Price History")

PERIOD_MAP = {"3Y": "3y", "1Y": "1y", "6M": "6mo", "YTD": "ytd"}
period_label = st.radio(
    "Price range",
    list(PERIOD_MAP.keys()),
    horizontal=True,
    key="price_period",
    label_visibility="collapsed",
)

history = fetch_price(ticker, PERIOD_MAP[period_label])
if history is not None and not history.empty:
    _display_chart(_line(history["Close"], f"{ticker} — {period_label}"), "price")
else:
    st.info("No price history available.")

# ── Full Financial Statements ───────────────────────────────────────

st.divider()
st.subheader("Full Financial Statements")
st.caption("Values in millions ($M). EPS in $/share. Tax Rate in %.")

with st.expander("Income Statement"):
    fin = fetch_financials(ticker)
    if fin is not None and not fin.empty:
        st.dataframe(_format_statement_display(fin), use_container_width=True)
    else:
        st.info("No income statement data available.")

with st.expander("Balance Sheet"):
    bal = fetch_balance(ticker)
    if bal is not None and not bal.empty:
        st.dataframe(_format_statement_display(bal), use_container_width=True)
    else:
        st.info("No balance sheet data available.")

with st.expander("Cash Flow Statement"):
    cf = fetch_cashflow(ticker)
    if cf is not None and not cf.empty:
        st.dataframe(_format_statement_display(cf), use_container_width=True)
    else:
        st.info("No cash flow data available.")
