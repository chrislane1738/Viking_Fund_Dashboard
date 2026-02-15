"""
Data Manager — abstraction layer for financial data fetching.

The frontend calls generic methods on a DataProvider interface.
Swap the concrete implementation (YFinanceProvider → AlphaVantageProvider)
without touching any UI code.
"""

from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import requests_cache

requests_cache.install_cache(
    "yfinance_cache",          # creates yfinance_cache.sqlite in working dir
    backend="sqlite",
    expire_after=3600,         # 1 hour — balances freshness vs. API usage
)


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
    """Return capital lease obligations with multi-level fallback.

    yfinance label availability varies by ticker and year:
      1. 'Capital Lease Obligations'              (preferred)
      2. 'Long Term Capital Lease Obligation' +
         'Current Capital Lease Obligation'        (component sum)
      3. 'Leases'  (right-of-use asset under PPE — close proxy when
         the liability labels are all NaN, e.g. AAPL 2024-2025)
    """
    primary = _safe_loc(bs, "Capital Lease Obligations")

    lt = _safe_loc(bs, "Long Term Capital Lease Obligation")
    ct = _safe_loc(bs, "Current Capital Lease Obligation")
    combined = lt.fillna(0) + ct.fillna(0)
    has_component = lt.notna() | ct.notna()
    combined = combined.where(has_component, other=float("nan"))

    leases_rou = _safe_loc(bs, "Leases")  # ROU asset fallback

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
    """Convert raw yfinance statement → $M values, date-only columns, ordered."""
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


# ── yfinance implementation ─────────────────────────────────────────

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
            # Valuation
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            # Dividend
            "dividend_yield": info.get("trailingAnnualDividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "ex_dividend_date": info.get("exDividendDate"),
            # Margins
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "ebitda_margin": info.get("ebitdaMargins"),
            # FCF yield inputs
            "current_price": info.get("currentPrice"),
            "free_cashflow": info.get("freeCashflow"),
            "shares_outstanding": info.get("sharesOutstanding"),
            # Net debt
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            # Logo
            "logo_url": f"https://financialmodelingprep.com/image-stock/{ticker}.png",
        }

    # ── Key metrics ─────────────────────────────────────────────

    def get_key_metrics(self, ticker: str, mode: str = "annual") -> pd.DataFrame:
        t = self._ticker(ticker)

        if mode == "ttm":
            return self._ttm_metrics(t)

        # Annual or Quarterly
        if mode == "quarterly":
            fins, bs, cf = (t.quarterly_financials,
                            t.quarterly_balance_sheet,
                            t.quarterly_cashflow)
        else:
            fins, bs, cf = t.financials, t.balance_sheet, t.cashflow

        if fins is None or fins.empty:
            return pd.DataFrame()

        # Limit annual to 3 years
        if mode == "annual":
            fins = fins.iloc[:, :3]
            if bs is not None and not bs.empty:
                bs = bs.iloc[:, :3]
            if cf is not None and not cf.empty:
                cf = cf.iloc[:, :3]

        # Align bs and cf columns to fins so all Series share the same index
        if bs is not None and not bs.empty:
            bs = bs.reindex(columns=fins.columns)
        if cf is not None and not cf.empty:
            cf = cf.reindex(columns=fins.columns)

        metrics = self._build_period_metrics(fins, bs, cf)

        # Format column labels
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
        """Build the standard metrics DataFrame from period-aligned statements."""
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

        # Round each row: EPS to 2 decimals, ratios to 2, everything else to 1
        for row in metrics.index:
            if row in ("Diluted EPS ($)", "Current Ratio", "Cash Ratio",
                       "D/A (%)", "D/E (%)"):
                metrics.loc[row] = metrics.loc[row].round(2)
            else:
                metrics.loc[row] = metrics.loc[row].round(1)

        # Sort columns descending (newest-first) for consistent ordering
        metrics = metrics.sort_index(axis=1, ascending=False)
        return metrics

    def _ttm_metrics(self, t) -> pd.DataFrame:
        """Compute trailing-twelve-month metrics from quarterly reports."""
        qf = t.quarterly_financials
        qbs = t.quarterly_balance_sheet
        qcf = t.quarterly_cashflow

        if qf is None or qf.shape[1] < 4:
            return pd.DataFrame()

        M = 1_000_000
        n_ttm = qf.shape[1] - 3  # number of TTM windows

        def flow_sum(df, label):
            """Sum a line item across the 4-quarter window."""
            if df is not None and label in df.index:
                return df.loc[label].sum()
            return float("nan")

        def stock_val(series, label):
            """Point-in-time value from the latest quarter-end balance sheet."""
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
            # Try primary label, then components, then ROU asset fallback
            _cl = stock_val(bs_snap, "Capital Lease Obligations")
            if pd.isna(_cl):
                _lt_cl = stock_val(bs_snap, "Long Term Capital Lease Obligation")
                _ct_cl = stock_val(bs_snap, "Current Capital Lease Obligation")
                if pd.notna(_lt_cl) or pd.notna(_ct_cl):
                    _cl = (0 if pd.isna(_lt_cl) else _lt_cl) + (0 if pd.isna(_ct_cl) else _ct_cl)
            if pd.isna(_cl):
                _cl = stock_val(bs_snap, "Leases")  # ROU asset proxy
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

    # ── Full statements ─────────────────────────────────────────

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


# ── Future: Alpha Vantage stub ──────────────────────────────────────
#
# class AlphaVantageProvider(DataProvider):
#     def __init__(self, api_key: str): ...
#     ...
#
# When ready, implement the same methods and swap the provider below.


# ── Factory / default provider ──────────────────────────────────────

def get_provider() -> DataProvider:
    """Return the active data provider. Change this one line to switch backends."""
    return YFinanceProvider()
