"""
Data Manager — abstraction layer for financial data fetching.

The frontend calls generic methods on a DataProvider interface.
Swap the concrete implementation (YFinanceProvider → FMPProvider)
without touching any UI code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import requests
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Guard optional yfinance imports (kept as fallback)
try:
    import yfinance as yf
    import requests_cache
    requests_cache.install_cache(
        "yfinance_cache",
        backend="sqlite",
        expire_after=3600,
    )
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


# ── Abstract interface ──────────────────────────────────────────────

class DataProvider(ABC):
    """Contract that every data backend must satisfy."""

    @abstractmethod
    def get_company_info(self, ticker: str) -> dict:
        """Return a dict of basic company metadata (name, sector, industry, etc.)."""

    @abstractmethod
    def get_key_metrics(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
        """Return a DataFrame of key fundamental metrics.
        mode: 'annual' | 'ttm' | 'quarterly'
        Columns = period labels, rows = metric names.
        Values in millions ($M) where applicable."""

    @abstractmethod
    def get_annual_financials(self, ticker: str) -> pd.DataFrame:
        """Return the annual income statement (values in $M, logical order)."""

    @abstractmethod
    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        """Return the quarterly income statement (values in $M, logical order)."""

    @abstractmethod
    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        """Return the balance sheet (values in $M)."""

    @abstractmethod
    def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        """Return the cash-flow statement (values in $M)."""

    @abstractmethod
    def get_price_history(self, ticker: str, period: str = "3y") -> pd.DataFrame:
        """Return historical price data (Date, Open, High, Low, Close, Volume)."""

    @abstractmethod
    def get_historical_ratios(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
        """Return a DataFrame of historical valuation ratios.
        mode: 'annual' | 'quarterly'
        Columns = period labels, rows = ratio names."""

    def get_quote(self, ticker: str) -> dict:
        """Return real-time quote data (price, change, open, previous close)."""
        return {}


# ── Constants ───────────────────────────────────────────────────────

# Rows that must NOT be divided by 1 M (already per-share or ratio)
_NO_DIVIDE = {"Diluted EPS", "Basic EPS", "Tax Rate For Calcs"}

# Preferred income-statement ordering (top → bottom)
_INCOME_ORDER = [
    "Total Revenue",
    "Operating Revenue",
    "Cost Of Revenue",
    "Gross Profit",
    "Research And Development",
    "Selling General And Administration",
    "Operating Expense",
    "Total Expenses",
    "Operating Income",
    "Total Operating Income As Reported",
    "EBIT",
    "EBITDA",
    "Normalized EBITDA",
    "Reconciled Depreciation",
    "Reconciled Cost Of Revenue",
    "Interest Income",
    "Interest Income Non Operating",
    "Interest Expense",
    "Interest Expense Non Operating",
    "Net Interest Income",
    "Net Non Operating Interest Income Expense",
    "Other Income Expense",
    "Other Non Operating Income Expenses",
    "Pretax Income",
    "Tax Provision",
    "Tax Rate For Calcs",
    "Tax Effect Of Unusual Items",
    "Net Income Continuous Operations",
    "Net Income Including Noncontrolling Interests",
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income From Continuing And Discontinued Operation",
    "Net Income From Continuing Operation Net Minority Interest",
    "Normalized Income",
    "Diluted NI Availto Com Stockholders",
    "Basic EPS",
    "Diluted EPS",
    "Basic Average Shares",
    "Diluted Average Shares",
]

# Non-dollar rows (no $ prefix when formatting)
_NO_DOLLAR = {
    "Tax Rate For Calcs",
    "Basic Average Shares", "Diluted Average Shares",
    "Ordinary Shares Number", "Share Issued", "Treasury Shares Number",
}

# Percentage rows
_PERCENT_ROWS = {"Tax Rate For Calcs"}

# Preferred balance-sheet ordering (top → bottom)
_BALANCE_ORDER = [
    # Assets
    "Current Assets", "Cash And Cash Equivalents", "Cash Equivalents",
    "Cash Financial", "Cash Cash Equivalents And Short Term Investments",
    "Other Short Term Investments", "Accounts Receivable", "Other Receivables",
    "Receivables", "Inventory", "Other Current Assets",
    # Non-Current Assets
    "Total Non Current Assets", "Net PPE", "Gross PPE",
    "Accumulated Depreciation", "Land And Improvements", "Properties",
    "Other Properties", "Machinery Furniture Equipment", "Leases",
    "Investmentin Financial Assets", "Available For Sale Securities",
    "Other Investments", "Investments And Advances",
    "Non Current Deferred Taxes Assets", "Non Current Deferred Assets",
    "Other Non Current Assets",
    # Total Assets
    "Total Assets",
    # Current Liabilities
    "Current Liabilities", "Accounts Payable", "Total Tax Payable",
    "Income Tax Payable", "Payables", "Payables And Accrued Expenses",
    "Current Accrued Expenses", "Current Debt", "Commercial Paper",
    "Other Current Borrowings", "Current Debt And Capital Lease Obligation",
    "Current Capital Lease Obligation", "Current Deferred Revenue",
    "Current Deferred Liabilities", "Other Current Liabilities",
    # Non-Current Liabilities
    "Total Non Current Liabilities Net Minority Interest", "Long Term Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Long Term Capital Lease Obligation", "Capital Lease Obligations",
    "Tradeand Other Payables Non Current", "Other Non Current Liabilities",
    # Total Liabilities
    "Total Liabilities Net Minority Interest",
    # Equity
    "Stockholders Equity", "Common Stock Equity",
    "Total Equity Gross Minority Interest", "Capital Stock", "Common Stock",
    "Retained Earnings", "Gains Losses Not Affecting Retained Earnings",
    "Other Equity Adjustments",
    # Derived
    "Total Capitalization", "Net Tangible Assets", "Tangible Book Value",
    "Invested Capital", "Working Capital", "Net Debt",
    "Share Issued", "Ordinary Shares Number", "Treasury Shares Number",
]

# Preferred cash-flow ordering (top → bottom)
_CASHFLOW_ORDER = [
    # Operating
    "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
    "Net Income From Continuing Operations", "Stock Based Compensation",
    "Depreciation Amortization Depletion", "Depreciation And Amortization",
    "Deferred Tax", "Deferred Income Tax", "Other Non Cash Items",
    "Change In Working Capital", "Change In Receivables",
    "Changes In Account Receivables", "Change In Inventory",
    "Change In Payables And Accrued Expense", "Change In Payable",
    "Change In Account Payable", "Change In Other Current Assets",
    "Change In Other Current Liabilities", "Change In Other Working Capital",
    # Investing
    "Investing Cash Flow", "Cash Flow From Continuing Investing Activities",
    "Capital Expenditure", "Purchase Of PPE", "Net PPE Purchase And Sale",
    "Purchase Of Investment", "Sale Of Investment",
    "Net Investment Purchase And Sale", "Purchase Of Business",
    "Net Business Purchase And Sale", "Net Other Investing Changes",
    # Financing
    "Financing Cash Flow", "Cash Flow From Continuing Financing Activities",
    "Common Stock Dividend Paid", "Cash Dividends Paid",
    "Repurchase Of Capital Stock", "Common Stock Issuance",
    "Common Stock Payments", "Net Common Stock Issuance",
    "Issuance Of Capital Stock", "Long Term Debt Issuance",
    "Long Term Debt Payments", "Net Long Term Debt Issuance",
    "Net Short Term Debt Issuance", "Net Issuance Payments Of Debt",
    "Issuance Of Debt", "Repayment Of Debt", "Net Other Financing Charges",
    # Summary
    "Free Cash Flow", "Interest Paid Supplemental Data",
    "Income Tax Paid Supplemental Data", "Changes In Cash",
    "Beginning Cash Position", "End Cash Position",
]


# ── Helpers ─────────────────────────────────────────────────────────

def _safe_loc(df: pd.DataFrame, label: str):
    """Return a row from df by label, or a Series of NaN if missing."""
    if df is not None and label in df.index:
        return df.loc[label]
    if df is not None:
        return pd.Series([float("nan")] * len(df.columns), index=df.columns)
    return pd.Series(dtype=float)


def _capital_lease_row(bs: pd.DataFrame):
    """Return capital lease obligations with multi-level fallback."""
    primary = _safe_loc(bs, "Capital Lease Obligations")
    lt = _safe_loc(bs, "Long Term Capital Lease Obligation")
    ct = _safe_loc(bs, "Current Capital Lease Obligation")
    combined = lt.fillna(0) + ct.fillna(0)
    has_component = lt.notna() | ct.notna()
    combined = combined.where(has_component, other=float("nan"))
    leases_rou = _safe_loc(bs, "Leases")
    return primary.fillna(combined).fillna(leases_rou)


def _quarter_label(dt) -> str:
    """Convert a datetime to 'Q1 '24' format."""
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} '{dt.strftime('%y')}"


def _reorder(df: pd.DataFrame, order: list) -> pd.DataFrame:
    """Reorder rows of df to match *order*; items not listed go to the end."""
    ordered = [r for r in order if r in df.index]
    remaining = [r for r in df.index if r not in ordered]
    return df.loc[ordered + remaining]


def _format_statement(df: pd.DataFrame, order: list | None = None) -> pd.DataFrame:
    """Convert raw statement → $M values, date-only columns, ordered."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for row in out.index:
        if row not in _NO_DIVIDE:
            out.loc[row] = out.loc[row] / 1_000_000
    out.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                   for c in out.columns]
    out = out.round(1)
    if order:
        out = _reorder(out, order)
    return out


def _format_statement_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a string-formatted copy with $ signs and % where applicable."""
    if df.empty:
        return df
    fmt = df.copy().astype(object)
    for row in fmt.index:
        for col in fmt.columns:
            val = df.loc[row, col]
            if pd.isna(val):
                fmt.loc[row, col] = "—"
            elif row in _PERCENT_ROWS:
                fmt.loc[row, col] = f"{val * 100:.1f}%" if abs(val) < 10 else f"{val:.1f}%"
            elif row in _NO_DOLLAR:
                fmt.loc[row, col] = f"{val:,.1f}"
            elif row in _NO_DIVIDE:
                fmt.loc[row, col] = f"${val:,.2f}"
            else:
                fmt.loc[row, col] = f"${val:,.1f}"
    return fmt


# ── FMP helpers ────────────────────────────────────────────────────

_FMP_BASE = "https://financialmodelingprep.com/stable"


def _get_fmp_key() -> str:
    """Load FMP API key from Streamlit secrets or environment."""
    try:
        return st.secrets["fmp"]["api_key"]
    except Exception:
        key = os.environ.get("FMP_API_KEY", "")
        if not key:
            raise ValueError("FMP API key not found in secrets or environment.")
        return key


def _fmp_get(endpoint: str, params: dict | None = None) -> list | dict:
    """Authenticated GET to FMP API. Returns parsed JSON."""
    params = params or {}
    params["apikey"] = _get_fmp_key()
    url = f"{_FMP_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _safe_get(d: dict | list, key: str, default=None):
    """Safely get a key from a dict; return default if d is a list or key missing."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


# ── FMP implementation ─────────────────────────────────────────────

class FMPProvider(DataProvider):

    # ── Company info ─────────────────────────────────────────────

    def get_company_info(self, ticker: str) -> dict:
        sym = {"symbol": ticker}

        profile = _fmp_get("profile", sym)
        if isinstance(profile, list) and profile:
            profile = profile[0]
        elif not isinstance(profile, dict):
            profile = {}

        ratios_ttm = _fmp_get("ratios-ttm", sym)
        if isinstance(ratios_ttm, list) and ratios_ttm:
            ratios_ttm = ratios_ttm[0]
        elif not isinstance(ratios_ttm, dict):
            ratios_ttm = {}

        metrics_ttm = _fmp_get("key-metrics-ttm", sym)
        if isinstance(metrics_ttm, list) and metrics_ttm:
            metrics_ttm = metrics_ttm[0]
        elif not isinstance(metrics_ttm, dict):
            metrics_ttm = {}

        bs = _fmp_get("balance-sheet-statement", {"symbol": ticker, "period": "quarter", "limit": 1})
        bs_latest = bs[0] if isinstance(bs, list) and bs else {}

        # Forward PE from analyst estimates
        forward_pe = None
        forward_peg = None
        try:
            estimates = _fmp_get("analyst-estimates", {"symbol": ticker, "period": "annual", "limit": 10})
            if isinstance(estimates, list) and estimates:
                today = datetime.now()
                # Find the nearest future fiscal year estimate with epsAvg
                for est in sorted(estimates, key=lambda e: e.get("date", ""), reverse=False):
                    est_date = datetime.strptime(est["date"], "%Y-%m-%d") if est.get("date") else None
                    forward_eps_avg = est.get("epsAvg")
                    if est_date and est_date > today and forward_eps_avg and forward_eps_avg != 0:
                        price_now = profile.get("price")
                        if price_now and price_now > 0:
                            forward_pe = price_now / forward_eps_avg
                            # Forward PEG: forward_pe / implied EPS growth rate
                            trailing_pe = ratios_ttm.get("priceToEarningsRatioTTM")
                            if trailing_pe and trailing_pe > 0:
                                current_eps = price_now / trailing_pe
                                if current_eps and abs(current_eps) > 0:
                                    growth_rate = ((forward_eps_avg - current_eps) / abs(current_eps)) * 100
                                    if growth_rate > 0:
                                        forward_peg = forward_pe / growth_rate
                        break
        except Exception:
            pass

        # Dividend date from dividends endpoint
        ex_dividend_date = None
        try:
            divs = _fmp_get("dividends", {"symbol": ticker})
            if isinstance(divs, list) and divs:
                ex_dividend_date = divs[0].get("date")
        except Exception:
            pass

        # FCF yield components
        price = profile.get("price")
        mktcap = profile.get("marketCap") or profile.get("mktCap")
        shares = int(mktcap / price) if price and mktcap and price > 0 else None
        fcf_ps = ratios_ttm.get("freeCashFlowPerShareTTM")
        free_cashflow = fcf_ps * shares if fcf_ps is not None and shares else None

        return {
            "name": profile.get("companyName", "N/A"),
            "sector": profile.get("sector", "N/A"),
            "industry": profile.get("industry", "N/A"),
            "market_cap": mktcap,
            "currency": profile.get("currency", "USD"),
            "summary": profile.get("description", ""),
            # Valuation
            "trailing_pe": ratios_ttm.get("priceToEarningsRatioTTM"),
            "forward_pe": forward_pe or ratios_ttm.get("priceToEarningsRatioTTM"),
            "price_to_sales": ratios_ttm.get("priceToSalesRatioTTM"),
            "price_to_book": ratios_ttm.get("priceToBookRatioTTM"),
            "ev_to_ebitda": ratios_ttm.get("enterpriseValueMultipleTTM"),
            "forward_peg": forward_peg,
            # Dividend
            "dividend_yield": ratios_ttm.get("dividendYieldTTM"),
            "payout_ratio": ratios_ttm.get("dividendPayoutRatioTTM"),
            "ex_dividend_date": ex_dividend_date,
            # Margins
            "operating_margin": ratios_ttm.get("operatingProfitMarginTTM"),
            "profit_margin": ratios_ttm.get("netProfitMarginTTM"),
            "ebitda_margin": ratios_ttm.get("ebitdaMarginTTM"),
            # FCF yield inputs
            "current_price": price,
            "free_cashflow": free_cashflow,
            "shares_outstanding": shares,
            # Net debt
            "total_cash": bs_latest.get("cashAndShortTermInvestments") or bs_latest.get("cashAndCashEquivalents"),
            "total_debt": bs_latest.get("totalDebt"),
            # Debt to equity
            "debt_to_equity": ratios_ttm.get("debtToEquityRatioTTM"),
            # Logo
            "logo_url": profile.get("image", f"https://financialmodelingprep.com/image-stock/{ticker}.png"),
        }

    # ── Key metrics ─────────────────────────────────────────────

    def get_key_metrics(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
        if mode == "ttm":
            return self._ttm_metrics(ticker)

        period = "quarter" if mode == "quarterly" else "annual"
        limit = 80 if mode == "quarterly" else 20

        inc = _fmp_get("income-statement", {"symbol": ticker, "period": period, "limit": limit})
        bs = _fmp_get("balance-sheet-statement", {"symbol": ticker, "period": period, "limit": limit})
        cf = _fmp_get("cash-flow-statement", {"symbol": ticker, "period": period, "limit": limit})

        if not inc:
            return pd.DataFrame()

        # Build dicts keyed by date string
        bs_map = {item["date"]: item for item in (bs or [])}
        cf_map = {item["date"]: item for item in (cf or [])}

        # FMP returns newest-first; build ordered list to preserve that
        col_order = []
        results = {}
        for item in inc:
            date_str = item["date"]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if mode == "annual":
                label = str(item.get("fiscalYear") or dt.year)
            else:
                # Use FMP's period field (Q1-Q4) with 2-digit year
                fmp_period = item.get("period", "")
                if fmp_period and fmp_period.startswith("Q"):
                    label = f"{fmp_period} '{dt.strftime('%y')}"
                else:
                    label = _quarter_label(dt)

            # Handle potential duplicate labels by appending date
            if label in results:
                label = f"{label} ({date_str})"

            col_order.append(label)
            b = bs_map.get(date_str, {})
            c = cf_map.get(date_str, {})

            results[label] = self._build_metric_row(item, b, c)

        metrics = pd.DataFrame(results)
        # Preserve FMP's newest-first ordering
        metrics = metrics.reindex(columns=col_order)
        return metrics

    def _build_metric_row(self, inc: dict, bs: dict, cf: dict) -> dict:
        """Build one column of key metrics from FMP statement dicts."""
        M = 1_000_000

        revenue = (inc.get("revenue") or 0) / M
        cogs = (inc.get("costOfRevenue") or 0) / M
        eps = inc.get("epsDiluted") or inc.get("epsdiluted")
        ebitda = (inc.get("ebitda") or 0) / M
        ebit = (inc.get("operatingIncome") or 0) / M
        da = (inc.get("depreciationAndAmortization") or 0) / M
        sbc = (cf.get("stockBasedCompensation") or 0) / M
        ocf = (cf.get("operatingCashFlow") or 0) / M
        fcf = (cf.get("freeCashFlow") or 0) / M
        rnd = (inc.get("researchAndDevelopmentExpenses") or 0) / M
        sga = (inc.get("sellingGeneralAndAdministrativeExpenses") or 0) / M
        ga = (inc.get("generalAndAdministrativeExpenses") or 0) / M
        sm = (inc.get("sellingAndMarketingExpenses") or 0) / M
        capex = abs(cf.get("capitalExpenditure") or 0) / M
        dividends = abs(cf.get("commonDividendsPaid") or cf.get("dividendsPaid") or 0) / M
        buybacks = abs(cf.get("commonStockRepurchased") or 0) / M
        shares = (inc.get("weightedAverageShsOutDil") or 0) / M

        total_cash = (bs.get("cashAndShortTermInvestments") or bs.get("cashAndCashEquivalents") or 0) / M
        total_debt = (bs.get("totalDebt") or 0) / M
        long_term_debt = (bs.get("longTermDebt") or 0) / M
        capital_lease = (bs.get("capitalLeaseObligations") or 0) / M

        equity = bs.get("totalStockholdersEquity") or 0
        total_assets = bs.get("totalAssets") or 0
        net_income = inc.get("netIncome") or 0
        current_assets = bs.get("totalCurrentAssets") or 0
        current_liabilities = bs.get("totalCurrentLiabilities") or 0
        cash_equiv = bs.get("cashAndShortTermInvestments") or bs.get("cashAndCashEquivalents") or 0
        td_raw = bs.get("totalDebt") or 0

        # Invested capital = equity + totalDebt - cash
        invested_capital = equity + td_raw - cash_equiv

        # Tax rate from income statement
        ibt = inc.get("incomeBeforeTax") or 0
        tax_expense = inc.get("incomeTaxExpense") or 0
        tax_rate = tax_expense / ibt if ibt != 0 else 0

        # NOPAT for ROIC
        op_income = inc.get("operatingIncome") or 0
        nopat = op_income * (1 - tax_rate)

        def safe_ratio(num, den):
            if den and den != 0:
                return round(num / den * 100, 2)
            return float("nan")

        def safe_div(num, den):
            if den and den != 0:
                return round(num / den, 2)
            return float("nan")

        raw_revenue = inc.get("revenue") or 0
        raw_op_income = inc.get("operatingIncome") or 0
        interest_expense_raw = inc.get("interestExpense") or 0
        raw_ebitda = inc.get("ebitda") or 0
        basic_eps = inc.get("eps")
        op_expenses = (inc.get("operatingExpenses") or 0) / M

        return {
            "Revenue ($M)": round(revenue, 1),
            "COGS ($M)": round(cogs, 1),
            "Diluted EPS ($)": round(eps, 2) if eps is not None else float("nan"),
            "Basic EPS ($)": round(basic_eps, 2) if basic_eps is not None else float("nan"),
            "EBITDA ($M)": round(ebitda, 1),
            "EBIT ($M)": round(ebit, 1),
            "D&A ($M)": round(da, 1),
            "Operating Income ($M)": round(ebit, 1),
            "Operating Expenses ($M)": round(op_expenses, 1),
            "SBC ($M)": round(sbc, 1),
            "OCF ($M)": round(ocf, 1),
            "FCF ($M)": round(fcf, 1),
            "Total Cash ($M)": round(total_cash, 1),
            "Total Debt ($M)": round(total_debt, 1),
            "Long Term Debt ($M)": round(long_term_debt, 1),
            "Capital Lease ($M)": round(capital_lease, 1),
            "ROIC (%)": safe_ratio(nopat, invested_capital),
            "ROE (%)": safe_ratio(net_income, equity),
            "ROA (%)": safe_ratio(net_income, total_assets),
            "Gross Margin (%)": safe_ratio(raw_revenue - (inc.get("costOfRevenue") or 0), raw_revenue),
            "Operating Margin (%)": safe_ratio(raw_op_income, raw_revenue),
            "Net Margin (%)": safe_ratio(net_income, raw_revenue),
            "EBITDA Margin (%)": safe_ratio(raw_ebitda, raw_revenue),
            "Current Ratio": safe_div(current_assets, current_liabilities),
            "Cash Ratio": safe_div(cash_equiv, current_liabilities),
            "D/A (%)": safe_ratio(td_raw, total_assets),
            "D/E (%)": safe_ratio(td_raw, equity),
            "Shares Outstanding (M)": round(shares, 1),
            "Dividends Paid ($M)": round(dividends, 1),
            "Buybacks ($M)": round(buybacks, 1),
            "R&D ($M)": round(rnd, 1),
            "SG&A ($M)": round(sga, 1),
            "G&A ($M)": round(ga, 1),
            "S&M ($M)": round(sm, 1),
            "CAPEX ($M)": round(capex, 1),
            "Interest Coverage": safe_div(raw_op_income, interest_expense_raw),
        }

    def _ttm_metrics(self, ticker: str) -> pd.DataFrame:
        """Compute trailing-twelve-month metrics from quarterly data."""
        inc = _fmp_get("income-statement", {"symbol": ticker, "period": "quarter", "limit": 83})
        bs = _fmp_get("balance-sheet-statement", {"symbol": ticker, "period": "quarter", "limit": 83})
        cf = _fmp_get("cash-flow-statement", {"symbol": ticker, "period": "quarter", "limit": 83})

        if not inc or len(inc) < 4:
            return pd.DataFrame()

        bs_map = {item["date"]: item for item in (bs or [])}
        cf_map = {item["date"]: item for item in (cf or [])}

        M = 1_000_000
        n_ttm = len(inc) - 3
        results = {}

        for i in range(n_ttm):
            window = inc[i:i + 4]
            end_date = datetime.strptime(window[0]["date"], "%Y-%m-%d")
            fmp_period = window[0].get("period", "")
            if fmp_period and fmp_period.startswith("Q"):
                label = f"{fmp_period} '{end_date.strftime('%y')}"
            else:
                label = _quarter_label(end_date)

            # Sum flow items across 4 quarters
            def flow_sum(items, key):
                return sum(item.get(key, 0) or 0 for item in items)

            def cf_flow_sum(key):
                vals = []
                for item in window:
                    c = cf_map.get(item["date"], {})
                    vals.append(c.get(key, 0) or 0)
                return sum(vals)

            # Balance sheet snapshot from the most recent quarter
            bs_snap = bs_map.get(window[0]["date"], {})

            revenue = flow_sum(window, "revenue") / M
            cogs = flow_sum(window, "costOfRevenue") / M
            eps = sum(item.get("epsDiluted") or item.get("epsdiluted") or 0 for item in window)
            ebitda = flow_sum(window, "ebitda") / M
            ebit_val = flow_sum(window, "operatingIncome") / M
            da = flow_sum(window, "depreciationAndAmortization") / M
            sbc = cf_flow_sum("stockBasedCompensation") / M
            ocf = cf_flow_sum("operatingCashFlow") / M
            fcf = cf_flow_sum("freeCashFlow") / M
            rnd = flow_sum(window, "researchAndDevelopmentExpenses") / M
            sga = flow_sum(window, "sellingGeneralAndAdministrativeExpenses") / M
            ga = flow_sum(window, "generalAndAdministrativeExpenses") / M
            sm = flow_sum(window, "sellingAndMarketingExpenses") / M
            capex = abs(cf_flow_sum("capitalExpenditure")) / M
            dividends = abs(cf_flow_sum("commonDividendsPaid") or cf_flow_sum("dividendsPaid")) / M
            buybacks = abs(cf_flow_sum("commonStockRepurchased")) / M
            shares = flow_sum(window, "weightedAverageShsOutDil") / 4 / M

            total_cash = (bs_snap.get("cashAndShortTermInvestments") or bs_snap.get("cashAndCashEquivalents") or 0) / M
            total_debt = (bs_snap.get("totalDebt") or 0) / M
            long_term_debt = (bs_snap.get("longTermDebt") or 0) / M
            capital_lease = (bs_snap.get("capitalLeaseObligations") or 0) / M

            ni = flow_sum(window, "netIncome")
            eq = bs_snap.get("totalStockholdersEquity") or 0
            ta = bs_snap.get("totalAssets") or 0
            td = bs_snap.get("totalDebt") or 0
            cash_raw = bs_snap.get("cashAndShortTermInvestments") or bs_snap.get("cashAndCashEquivalents") or 0
            ic = eq + td - cash_raw
            op_income = flow_sum(window, "operatingIncome")
            cur_assets = bs_snap.get("totalCurrentAssets") or 0
            cur_liab = bs_snap.get("totalCurrentLiabilities") or 0

            # Tax rate: average across quarters
            ibt = flow_sum(window, "incomeBeforeTax")
            tax_exp = flow_sum(window, "incomeTaxExpense")
            tax_rate = tax_exp / ibt if ibt != 0 else 0
            nopat = op_income * (1 - tax_rate)

            def safe_ratio(num, den):
                if den and den != 0:
                    return round(num / den * 100, 2)
                return float("nan")

            def safe_div(num, den):
                if den and den != 0:
                    return round(num / den, 2)
                return float("nan")

            int_exp_ttm = flow_sum(window, "interestExpense")
            basic_eps_ttm = sum(item.get("eps") or 0 for item in window)
            op_expenses_ttm = flow_sum(window, "operatingExpenses") / M
            raw_rev = flow_sum(window, "revenue")
            raw_cogs_ttm = flow_sum(window, "costOfRevenue")
            raw_ebitda_ttm = flow_sum(window, "ebitda")

            results[label] = {
                "Revenue ($M)": round(revenue, 1),
                "COGS ($M)": round(cogs, 1),
                "Diluted EPS ($)": round(eps, 2) if pd.notna(eps) else float("nan"),
                "Basic EPS ($)": round(basic_eps_ttm, 2),
                "EBITDA ($M)": round(ebitda, 1),
                "EBIT ($M)": round(ebit_val, 1),
                "D&A ($M)": round(da, 1),
                "Operating Income ($M)": round(ebit_val, 1),
                "Operating Expenses ($M)": round(op_expenses_ttm, 1),
                "SBC ($M)": round(sbc, 1),
                "OCF ($M)": round(ocf, 1),
                "FCF ($M)": round(fcf, 1),
                "Total Cash ($M)": round(total_cash, 1),
                "Total Debt ($M)": round(total_debt, 1),
                "Long Term Debt ($M)": round(long_term_debt, 1),
                "Capital Lease ($M)": round(capital_lease, 1),
                "ROIC (%)": safe_ratio(nopat, ic),
                "ROE (%)": safe_ratio(ni, eq),
                "ROA (%)": safe_ratio(ni, ta),
                "Gross Margin (%)": safe_ratio(raw_rev - raw_cogs_ttm, raw_rev),
                "Operating Margin (%)": safe_ratio(op_income, raw_rev),
                "Net Margin (%)": safe_ratio(ni, raw_rev),
                "EBITDA Margin (%)": safe_ratio(raw_ebitda_ttm, raw_rev),
                "Current Ratio": safe_div(cur_assets, cur_liab),
                "Cash Ratio": safe_div(cash_raw, cur_liab),
                "D/A (%)": safe_ratio(td, ta),
                "D/E (%)": safe_ratio(td, eq),
                "Shares Outstanding (M)": round(shares, 1),
                "Dividends Paid ($M)": round(dividends, 1),
                "Buybacks ($M)": round(buybacks, 1),
                "R&D ($M)": round(rnd, 1),
                "SG&A ($M)": round(sga, 1),
                "G&A ($M)": round(ga, 1),
                "S&M ($M)": round(sm, 1),
                "CAPEX ($M)": round(capex, 1),
                "Interest Coverage": safe_div(op_income, int_exp_ttm),
            }

        return pd.DataFrame(results)

    # ── Full income statement ────────────────────────────────────

    def get_annual_financials(self, ticker: str) -> pd.DataFrame:
        return self._build_income_statement(ticker, "annual", 20)

    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        return self._build_income_statement(ticker, "quarter", 40)

    def _build_income_statement(self, ticker: str, period: str, limit: int) -> pd.DataFrame:
        data = _fmp_get("income-statement", {"symbol": ticker, "period": period, "limit": limit})
        if not data:
            return pd.DataFrame()

        # FMP field -> display row name
        field_map = {
            "revenue": "Total Revenue",
            "costOfRevenue": "Cost Of Revenue",
            "grossProfit": "Gross Profit",
            "researchAndDevelopmentExpenses": "Research And Development",
            "sellingGeneralAndAdministrativeExpenses": "Selling General And Administration",
            "operatingExpenses": "Operating Expense",
            "operatingIncome": "Operating Income",
            "ebitda": "EBITDA",
            "depreciationAndAmortization": "Reconciled Depreciation",
            "interestIncome": "Interest Income",
            "interestExpense": "Interest Expense",
            "incomeBeforeTax": "Pretax Income",
            "incomeTaxExpense": "Tax Provision",
            "netIncome": "Net Income",
            "netIncomeDeprec": "Net Income Common Stockholders",
            "eps": "Basic EPS",
            "epsDiluted": "Diluted EPS",
            "weightedAverageShsOut": "Basic Average Shares",
            "weightedAverageShsOutDil": "Diluted Average Shares",
        }

        rows = {}
        dates = []
        for item in data:
            date_str = item["date"]
            dates.append(date_str)
            for fmp_key, row_name in field_map.items():
                val = item.get(fmp_key)
                if val is not None:
                    rows.setdefault(row_name, {})[date_str] = val

            # Derived: Tax Rate For Calcs
            ibt = item.get("incomeBeforeTax") or 0
            tax = item.get("incomeTaxExpense") or 0
            tax_rate = tax / ibt if ibt != 0 else 0
            rows.setdefault("Tax Rate For Calcs", {})[date_str] = tax_rate

            # Net Income Common Stockholders (use netIncome if dedicated field absent)
            if "Net Income Common Stockholders" not in rows or date_str not in rows.get("Net Income Common Stockholders", {}):
                rows.setdefault("Net Income Common Stockholders", {})[date_str] = item.get("netIncome")

        df = pd.DataFrame(rows).T
        # Ensure column order matches date order
        df = df.reindex(columns=dates)

        return _format_statement(df, order=_INCOME_ORDER)

    # ── Balance sheet ────────────────────────────────────────────

    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        period = "quarter" if quarterly else "annual"
        limit = 40 if quarterly else 20
        data = _fmp_get("balance-sheet-statement", {"symbol": ticker, "period": period, "limit": limit})
        if not data:
            return pd.DataFrame()

        field_map = {
            "totalCurrentAssets": "Current Assets",
            "cashAndCashEquivalents": "Cash And Cash Equivalents",
            "cashAndShortTermInvestments": "Cash Cash Equivalents And Short Term Investments",
            "shortTermInvestments": "Other Short Term Investments",
            "netReceivables": "Receivables",
            "inventory": "Inventory",
            "otherCurrentAssets": "Other Current Assets",
            "totalNonCurrentAssets": "Total Non Current Assets",
            "propertyPlantEquipmentNet": "Net PPE",
            "longTermInvestments": "Investmentin Financial Assets",
            "otherNonCurrentAssets": "Other Non Current Assets",
            "totalAssets": "Total Assets",
            "totalCurrentLiabilities": "Current Liabilities",
            "accountPayables": "Accounts Payable",
            "shortTermDebt": "Current Debt",
            "otherCurrentLiabilities": "Other Current Liabilities",
            "totalNonCurrentLiabilities": "Total Non Current Liabilities Net Minority Interest",
            "longTermDebt": "Long Term Debt",
            "capitalLeaseObligations": "Capital Lease Obligations",
            "otherNonCurrentLiabilities": "Other Non Current Liabilities",
            "totalLiabilities": "Total Liabilities Net Minority Interest",
            "totalStockholdersEquity": "Stockholders Equity",
            "commonStock": "Common Stock",
            "retainedEarnings": "Retained Earnings",
            "totalEquity": "Total Equity Gross Minority Interest",
            "totalDebt": "Total Debt",  # mapped to display, but won't auto-order
            "netDebt": "Net Debt",
        }

        rows = {}
        dates = []
        for item in data:
            date_str = item["date"]
            dates.append(date_str)
            for fmp_key, row_name in field_map.items():
                val = item.get(fmp_key)
                if val is not None:
                    rows.setdefault(row_name, {})[date_str] = val

            # Derived fields
            equity = item.get("totalStockholdersEquity") or 0
            debt = item.get("totalDebt") or 0
            cash = item.get("cashAndShortTermInvestments") or item.get("cashAndCashEquivalents") or 0
            rows.setdefault("Invested Capital", {})[date_str] = equity + debt - cash
            rows.setdefault("Working Capital", {})[date_str] = (item.get("totalCurrentAssets") or 0) - (item.get("totalCurrentLiabilities") or 0)

        df = pd.DataFrame(rows).T
        df = df.reindex(columns=dates)
        return _format_statement(df, order=_BALANCE_ORDER)

    # ── Cash flow ────────────────────────────────────────────────

    def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        period = "quarter" if quarterly else "annual"
        limit = 40 if quarterly else 20
        data = _fmp_get("cash-flow-statement", {"symbol": ticker, "period": period, "limit": limit})
        if not data:
            return pd.DataFrame()

        field_map = {
            "operatingCashFlow": "Operating Cash Flow",
            "netIncome": "Net Income From Continuing Operations",
            "stockBasedCompensation": "Stock Based Compensation",
            "depreciationAndAmortization": "Depreciation Amortization Depletion",
            "deferredIncomeTax": "Deferred Income Tax",
            "otherNonCashItems": "Other Non Cash Items",
            "changeInWorkingCapital": "Change In Working Capital",
            "accountsReceivables": "Changes In Account Receivables",
            "inventory": "Change In Inventory",
            "accountsPayables": "Change In Account Payable",
            "otherWorkingCapital": "Change In Other Working Capital",
            "capitalExpenditure": "Capital Expenditure",
            "acquisitionsNet": "Purchase Of Business",
            "purchasesOfInvestments": "Purchase Of Investment",
            "salesMaturitiesOfInvestments": "Sale Of Investment",
            "otherInvestingActivities": "Net Other Investing Changes",
            "commonDividendsPaid": "Common Stock Dividend Paid",
            "commonStockRepurchased": "Repurchase Of Capital Stock",
            "commonStockIssuance": "Common Stock Issuance",
            "debtRepayment": "Repayment Of Debt",
            "otherFinancingActivities": "Net Other Financing Charges",
            "freeCashFlow": "Free Cash Flow",
            "netChangeInCash": "Changes In Cash",
        }

        rows = {}
        dates = []
        for item in data:
            date_str = item["date"]
            dates.append(date_str)
            for fmp_key, row_name in field_map.items():
                val = item.get(fmp_key)
                if val is not None:
                    rows.setdefault(row_name, {})[date_str] = val

            # Use FMP's native totals if available, otherwise compute
            inv_cf = item.get("netCashProvidedByInvestingActivities")
            if inv_cf is None:
                inv_cf = (item.get("capitalExpenditure") or 0) + \
                         (item.get("acquisitionsNet") or 0) + \
                         (item.get("purchasesOfInvestments") or 0) + \
                         (item.get("salesMaturitiesOfInvestments") or 0) + \
                         (item.get("otherInvestingActivities") or 0)
            rows.setdefault("Investing Cash Flow", {})[date_str] = inv_cf

            fin_cf = item.get("netCashProvidedByFinancingActivities")
            if fin_cf is None:
                fin_cf = (item.get("commonDividendsPaid") or 0) + \
                         (item.get("commonStockRepurchased") or 0) + \
                         (item.get("commonStockIssuance") or 0) + \
                         (item.get("debtRepayment") or 0) + \
                         (item.get("otherFinancingActivities") or 0)
            rows.setdefault("Financing Cash Flow", {})[date_str] = fin_cf

        df = pd.DataFrame(rows).T
        df = df.reindex(columns=dates)
        return _format_statement(df, order=_CASHFLOW_ORDER)

    # ── Price history ────────────────────────────────────────────

    def get_price_history(self, ticker: str, period: str = "3y") -> pd.DataFrame:
        today = datetime.now()
        period_map = {
            "10y": timedelta(days=10 * 365),
            "5y": timedelta(days=5 * 365),
            "3y": timedelta(days=3 * 365),
            "1y": timedelta(days=365),
            "6mo": timedelta(days=182),
            "3mo": timedelta(days=91),
        }

        if period == "ytd":
            start = datetime(today.year, 1, 1)
        else:
            delta = period_map.get(period, timedelta(days=3 * 365))
            start = today - delta

        from_date = start.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        data = _fmp_get("historical-price-eod/full", {"symbol": ticker, "from": from_date, "to": to_date})

        historicals = data if isinstance(data, list) else data.get("historical", []) if isinstance(data, dict) else []
        if not historicals:
            return pd.DataFrame()

        df = pd.DataFrame(historicals)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")

        # Rename columns to match yfinance convention
        col_map = {"open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ── Historical ratios ──────────────────────────────────────────

    def get_historical_ratios(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
        period = "quarter" if mode == "quarterly" else "annual"
        limit = 40 if mode == "quarterly" else 10
        data = _fmp_get("ratios", {"symbol": ticker, "period": period, "limit": limit})
        if not data:
            return pd.DataFrame()

        # For quarterly mode, FMP's P/S and EV/EBITDA use single-quarter
        # denominators. Compute them from scratch using market cap, EV,
        # and rolling 4-quarter revenue / EBITDA.
        ttm_ps_ev = {}
        if mode == "quarterly":
            inc = _fmp_get("income-statement",
                           {"symbol": ticker, "period": "quarter",
                            "limit": limit + 3})
            km = _fmp_get("key-metrics",
                          {"symbol": ticker, "period": "quarter",
                           "limit": limit})
            # Build market-cap / EV lookup keyed by date
            km_map = {}
            if km:
                for k in km:
                    km_map[k.get("date")] = k
            # Build rolling TTM revenue & EBITDA, then compute ratios
            if inc and len(inc) >= 4:
                # inc is newest-first; build TTM windows
                for i in range(len(inc) - 3):
                    window = inc[i:i + 4]
                    q_date = window[0]["date"]
                    ttm_rev = sum((w.get("revenue") or 0) for w in window)
                    ttm_ebitda = sum((w.get("ebitda") or 0) for w in window)
                    k = km_map.get(q_date, {})
                    mcap = k.get("marketCap")
                    ev = k.get("enterpriseValue")
                    ps_ttm = None
                    ev_ebitda_ttm = None
                    if mcap and ttm_rev and ttm_rev != 0:
                        ps_ttm = mcap / ttm_rev
                    if ev and ttm_ebitda and ttm_ebitda != 0:
                        ev_ebitda_ttm = ev / ttm_ebitda
                    ttm_ps_ev[q_date] = {"ps": ps_ttm, "ev_ebitda": ev_ebitda_ttm}

        col_order = []
        results = {}
        for item in data:
            date_str = item.get("date", "")
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if mode == "quarterly":
                fmp_period = item.get("period", "")
                if fmp_period and fmp_period.startswith("Q"):
                    label = f"{fmp_period} '{dt.strftime('%y')}"
                else:
                    label = _quarter_label(dt)
            else:
                label = str(dt.year)

            if label in results:
                label = f"{label} ({date_str})"

            ps = item.get("priceToSalesRatio")
            ev_ebitda = item.get("enterpriseValueMultiple")

            # Override with properly computed TTM P/S and EV/EBITDA
            if mode == "quarterly" and date_str in ttm_ps_ev:
                computed = ttm_ps_ev[date_str]
                if computed["ps"] is not None:
                    ps = computed["ps"]
                if computed["ev_ebitda"] is not None:
                    ev_ebitda = computed["ev_ebitda"]

            col_order.append(label)
            results[label] = {
                "P/E Ratio": item.get("priceToEarningsRatio"),
                "P/S Ratio": ps,
                "P/B Ratio": item.get("priceToBookRatio"),
                "EV/EBITDA": ev_ebitda,
                "P/FCF Ratio": item.get("priceToFreeCashFlowRatio"),
                "Debt/Equity": item.get("debtToEquityRatio"),
                "Current Ratio": item.get("currentRatio"),
                "Dividend Yield (%)": round((item.get("dividendYield") or 0) * 100, 2),
            }

        df = pd.DataFrame(results)
        df = df.reindex(columns=col_order)
        return df

    def get_quote(self, ticker: str) -> dict:
        data = _fmp_get("quote", {"symbol": ticker})
        q = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        return {
            "price": q.get("price"),
            "change": q.get("change"),
            "change_pct": q.get("changesPercentage"),
            "open": q.get("open"),
            "previous_close": q.get("previousClose"),
            "name": q.get("name"),
        }


# ── yfinance implementation (kept as fallback) ─────────────────────

if _HAS_YFINANCE:
    class YFinanceProvider(DataProvider):

        def _ticker(self, ticker: str) -> yf.Ticker:
            return yf.Ticker(ticker)

        def get_company_info(self, ticker: str) -> dict:
            info = self._ticker(ticker).info
            return {
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
                "summary": info.get("longBusinessSummary", ""),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "price_to_book": info.get("priceToBook"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "forward_peg": None,
                "debt_to_equity": info.get("debtToEquity"),
                "dividend_yield": info.get("trailingAnnualDividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": info.get("exDividendDate"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
                "current_price": info.get("currentPrice"),
                "free_cashflow": info.get("freeCashflow"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "logo_url": f"https://financialmodelingprep.com/image-stock/{ticker}.png",
            }

        def get_key_metrics(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
            t = self._ticker(ticker)
            if mode == "ttm":
                return self._ttm_metrics(t)
            if mode == "quarterly":
                fins, bs, cf = (t.quarterly_financials,
                                t.quarterly_balance_sheet,
                                t.quarterly_cashflow)
            else:
                fins, bs, cf = t.financials, t.balance_sheet, t.cashflow
            if fins is None or fins.empty:
                return pd.DataFrame()
            if mode == "annual":
                fins = fins.iloc[:, :3]
                if bs is not None and not bs.empty:
                    bs = bs.iloc[:, :3]
                if cf is not None and not cf.empty:
                    cf = cf.iloc[:, :3]
            if bs is not None and not bs.empty:
                bs = bs.reindex(columns=fins.columns)
            if cf is not None and not cf.empty:
                cf = cf.reindex(columns=fins.columns)
            metrics = self._build_period_metrics(fins, bs, cf)
            new_cols = []
            for dt in metrics.columns:
                if hasattr(dt, "year"):
                    new_cols.append(str(dt.year) if mode == "annual"
                                    else _quarter_label(dt))
                else:
                    new_cols.append(str(dt))
            metrics.columns = new_cols
            return metrics

        def _build_period_metrics(self, fins, bs, cf) -> pd.DataFrame:
            M = 1_000_000
            revenue = _safe_loc(fins, "Total Revenue") / M
            cogs = _safe_loc(fins, "Cost Of Revenue") / M
            diluted_eps = _safe_loc(fins, "Diluted EPS")
            sbc = _safe_loc(cf, "Stock Based Compensation") / M
            ocf = _safe_loc(cf, "Operating Cash Flow") / M
            fcf = _safe_loc(cf, "Free Cash Flow") / M
            total_cash = _safe_loc(bs, "Cash Cash Equivalents And Short Term Investments") / M
            total_debt = _safe_loc(bs, "Total Debt") / M
            long_term_debt = _safe_loc(bs, "Long Term Debt") / M
            capital_lease = _capital_lease_row(bs) / M
            shares = _safe_loc(fins, "Diluted Average Shares") / M
            dividends = -_safe_loc(cf, "Common Stock Dividend Paid") / M
            buybacks = -_safe_loc(cf, "Repurchase Of Capital Stock") / M
            ebitda = _safe_loc(fins, "EBITDA") / M
            ebit = _safe_loc(fins, "EBIT") / M
            da = _safe_loc(fins, "Reconciled Depreciation") / M
            rnd = _safe_loc(fins, "Research And Development") / M
            sga = _safe_loc(fins, "Selling General And Administration") / M
            capex = -_safe_loc(cf, "Capital Expenditure") / M
            net_income = _safe_loc(fins, "Net Income")
            equity = _safe_loc(bs, "Stockholders Equity")
            total_assets = _safe_loc(bs, "Total Assets")
            invested_capital = _safe_loc(bs, "Invested Capital")
            roe = (net_income / equity * 100).round(2)
            roa = (net_income / total_assets * 100).round(2)
            roic = (_safe_loc(fins, "EBIT")
                    * (1 - _safe_loc(fins, "Tax Rate For Calcs"))
                    / invested_capital * 100).round(2)
            current_assets = _safe_loc(bs, "Current Assets")
            current_liabilities = _safe_loc(bs, "Current Liabilities")
            cash_and_equiv = _safe_loc(bs, "Cash Cash Equivalents And Short Term Investments")
            current_ratio = (current_assets / current_liabilities).round(2)
            cash_ratio = (cash_and_equiv / current_liabilities).round(2)
            debt_to_assets = ((_safe_loc(bs, "Total Debt") / total_assets) * 100).round(2)
            debt_to_equity = ((_safe_loc(bs, "Total Debt") / equity) * 100).round(2)
            metrics = pd.DataFrame({
                "Revenue ($M)": revenue,
                "COGS ($M)": cogs,
                "Diluted EPS ($)": diluted_eps,
                "EBITDA ($M)": ebitda,
                "EBIT ($M)": ebit,
                "D&A ($M)": da,
                "SBC ($M)": sbc,
                "OCF ($M)": ocf,
                "FCF ($M)": fcf,
                "Total Cash ($M)": total_cash,
                "Total Debt ($M)": total_debt,
                "Long Term Debt ($M)": long_term_debt,
                "Capital Lease ($M)": capital_lease,
                "ROIC (%)": roic,
                "ROE (%)": roe,
                "ROA (%)": roa,
                "Current Ratio": current_ratio,
                "Cash Ratio": cash_ratio,
                "D/A (%)": debt_to_assets,
                "D/E (%)": debt_to_equity,
                "Shares Outstanding (M)": shares,
                "Dividends Paid ($M)": dividends,
                "Buybacks ($M)": buybacks,
                "R&D ($M)": rnd,
                "SG&A ($M)": sga,
                "CAPEX ($M)": capex,
            }).T
            for row in metrics.index:
                if row in ("Diluted EPS ($)", "Current Ratio", "Cash Ratio",
                           "D/A (%)", "D/E (%)"):
                    metrics.loc[row] = metrics.loc[row].round(2)
                else:
                    metrics.loc[row] = metrics.loc[row].round(1)
            metrics = metrics.sort_index(axis=1, ascending=False)
            return metrics

        def _ttm_metrics(self, t) -> pd.DataFrame:
            qf = t.quarterly_financials
            qbs = t.quarterly_balance_sheet
            qcf = t.quarterly_cashflow
            if qf is None or qf.shape[1] < 4:
                return pd.DataFrame()
            M = 1_000_000
            n_ttm = qf.shape[1] - 3

            def flow_sum(df, label):
                if df is not None and label in df.index:
                    return df.loc[label].sum()
                return float("nan")

            def stock_val(series, label):
                if series is not None and label in series.index:
                    return series[label]
                return float("nan")

            def safe_ratio(num, den):
                if pd.notna(num) and pd.notna(den) and den != 0:
                    return round(num / den * 100, 2)
                return float("nan")

            results = {}
            for i in range(n_ttm):
                end_date = qf.columns[i]
                qf_w = qf.iloc[:, i:i + 4]
                qcf_w = (qcf.iloc[:, i:i + 4]
                         if qcf is not None and qcf.shape[1] > i + 3
                         else None)
                bs_snap = (qbs.iloc[:, i]
                           if qbs is not None and qbs.shape[1] > i
                           else None)
                revenue = flow_sum(qf_w, "Total Revenue") / M
                cogs = flow_sum(qf_w, "Cost Of Revenue") / M
                eps = flow_sum(qf_w, "Diluted EPS")
                ebitda = flow_sum(qf_w, "EBITDA") / M
                ebit_val = flow_sum(qf_w, "EBIT") / M
                da = flow_sum(qf_w, "Reconciled Depreciation") / M
                sbc = flow_sum(qcf_w, "Stock Based Compensation") / M
                ocf = flow_sum(qcf_w, "Operating Cash Flow") / M
                fcf = flow_sum(qcf_w, "Free Cash Flow") / M
                dividends = -flow_sum(qcf_w, "Common Stock Dividend Paid") / M
                buybacks = -flow_sum(qcf_w, "Repurchase Of Capital Stock") / M
                rnd = flow_sum(qf_w, "Research And Development") / M
                sga = flow_sum(qf_w, "Selling General And Administration") / M
                capex = -flow_sum(qcf_w, "Capital Expenditure") / M
                total_cash = stock_val(bs_snap, "Cash Cash Equivalents And Short Term Investments") / M
                total_debt = stock_val(bs_snap, "Total Debt") / M
                long_term_debt = stock_val(bs_snap, "Long Term Debt") / M
                _cl = stock_val(bs_snap, "Capital Lease Obligations")
                if pd.isna(_cl):
                    _lt_cl = stock_val(bs_snap, "Long Term Capital Lease Obligation")
                    _ct_cl = stock_val(bs_snap, "Current Capital Lease Obligation")
                    if pd.notna(_lt_cl) or pd.notna(_ct_cl):
                        _cl = (0 if pd.isna(_lt_cl) else _lt_cl) + (0 if pd.isna(_ct_cl) else _ct_cl)
                if pd.isna(_cl):
                    _cl = stock_val(bs_snap, "Leases")
                capital_lease = _cl / M
                shares = flow_sum(qf_w, "Diluted Average Shares") / 4 / M
                cur_assets = stock_val(bs_snap, "Current Assets")
                cur_liabilities = stock_val(bs_snap, "Current Liabilities")
                cash_equiv = stock_val(bs_snap, "Cash Cash Equivalents And Short Term Investments")
                ni = flow_sum(qf_w, "Net Income")
                eq = stock_val(bs_snap, "Stockholders Equity")
                ta = stock_val(bs_snap, "Total Assets")
                td = stock_val(bs_snap, "Total Debt")
                ic = stock_val(bs_snap, "Invested Capital")
                ebit = flow_sum(qf_w, "EBIT")
                tax_rate = 0.0
                if qf_w is not None and "Tax Rate For Calcs" in qf_w.index:
                    tax_rate = qf_w.loc["Tax Rate For Calcs"].mean()
                nopat = ebit * (1 - tax_rate) if pd.notna(ebit) else float("nan")
                label = _quarter_label(end_date)
                results[label] = {
                    "Revenue ($M)": round(revenue, 1),
                    "COGS ($M)": round(cogs, 1),
                    "Diluted EPS ($)": round(eps, 2) if pd.notna(eps) else float("nan"),
                    "EBITDA ($M)": round(ebitda, 1),
                    "EBIT ($M)": round(ebit_val, 1),
                    "D&A ($M)": round(da, 1),
                    "SBC ($M)": round(sbc, 1),
                    "OCF ($M)": round(ocf, 1),
                    "FCF ($M)": round(fcf, 1),
                    "Total Cash ($M)": round(total_cash, 1),
                    "Total Debt ($M)": round(total_debt, 1),
                    "Long Term Debt ($M)": round(long_term_debt, 1) if pd.notna(long_term_debt) else float("nan"),
                    "Capital Lease ($M)": round(capital_lease, 1) if pd.notna(capital_lease) else float("nan"),
                    "ROIC (%)": safe_ratio(nopat, ic),
                    "ROE (%)": safe_ratio(ni, eq),
                    "ROA (%)": safe_ratio(ni, ta),
                    "Current Ratio": round(cur_assets / cur_liabilities, 2) if pd.notna(cur_assets) and pd.notna(cur_liabilities) and cur_liabilities != 0 else float("nan"),
                    "Cash Ratio": round(cash_equiv / cur_liabilities, 2) if pd.notna(cash_equiv) and pd.notna(cur_liabilities) and cur_liabilities != 0 else float("nan"),
                    "D/A (%)": safe_ratio(td, ta),
                    "D/E (%)": safe_ratio(td, eq),
                    "Shares Outstanding (M)": round(shares, 1),
                    "Dividends Paid ($M)": round(dividends, 1),
                    "Buybacks ($M)": round(buybacks, 1),
                    "R&D ($M)": round(rnd, 1),
                    "SG&A ($M)": round(sga, 1),
                    "CAPEX ($M)": round(capex, 1),
                }
            return pd.DataFrame(results)

        def get_annual_financials(self, ticker: str) -> pd.DataFrame:
            return _format_statement(self._ticker(ticker).financials, order=_INCOME_ORDER)

        def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
            return _format_statement(self._ticker(ticker).quarterly_financials, order=_INCOME_ORDER)

        def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
            t = self._ticker(ticker)
            raw = t.quarterly_balance_sheet if quarterly else t.balance_sheet
            return _format_statement(raw, order=_BALANCE_ORDER)

        def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
            t = self._ticker(ticker)
            raw = t.quarterly_cashflow if quarterly else t.cashflow
            return _format_statement(raw, order=_CASHFLOW_ORDER)

        def get_price_history(self, ticker: str, period: str = "3y") -> pd.DataFrame:
            return self._ticker(ticker).history(period=period)

        def get_historical_ratios(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
            return pd.DataFrame()  # Not implemented for yfinance fallback


# ── Earnings calendar (standalone, not tied to a provider) ──────────

def fetch_earnings_calendar(from_date: str, to_date: str) -> list[dict]:
    """Fetch upcoming earnings for a date range from FMP."""
    try:
        data = _fmp_get("earnings-calendar", {"from": from_date, "to": to_date})
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_stock_news(ticker: str, limit: int = 10) -> list[dict]:
    """Fetch recent news for a ticker from FMP."""
    try:
        data = _fmp_get("news/stock", {"symbols": ticker, "limit": limit})
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_ticker_earnings(ticker: str, limit: int = 4) -> list[dict]:
    """Fetch earnings history/upcoming for a specific ticker from FMP."""
    try:
        data = _fmp_get("earnings", {"symbol": ticker, "limit": limit})
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ── FRED helpers ──────────────────────────────────────────────────────

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _get_fred_key() -> str:
    """Load FRED API key from Streamlit secrets or environment."""
    try:
        return st.secrets["fred"]["api_key"]
    except Exception:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            raise ValueError("FRED API key not found in secrets or environment.")
        return key


def fetch_fred_series(series_id, observation_start=None, observation_end=None):
    """Fetch FRED series → DataFrame with DatetimeIndex + float 'value' column.
    Drops missing values (FRED returns '.' for gaps)."""
    try:
        params = {
            "series_id": series_id,
            "api_key": _get_fred_key(),
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        resp = requests.get(_FRED_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["value"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df[["value"]]
    except Exception:
        return pd.DataFrame(columns=["value"])


# ── FMP macro helpers ─────────────────────────────────────────────────

def fetch_treasury_rates(from_date: str, to_date: str) -> list[dict]:
    """FMP treasury-rates → list of dicts with month1..year30 fields."""
    try:
        data = _fmp_get("treasury-rates", {"from": from_date, "to": to_date})
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_econ_calendar(from_date: str, to_date: str) -> list[dict]:
    """FMP economic-calendar → list of dicts with event/date/country/actual/estimate/previous/impact."""
    try:
        data = _fmp_get("economic-calendar", {"from": from_date, "to": to_date})
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ── Factory / default provider ──────────────────────────────────────

def get_provider() -> DataProvider:
    """Return the active data provider. Change this one line to switch backends."""
    return FMPProvider()
    # return YFinanceProvider()  # uncomment to revert to yfinance
