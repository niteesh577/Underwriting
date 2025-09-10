from langsmith import traceable
from typing import List, Dict, Any, Optional
import pandas as pd
from utils.helpers import _clean_dataframe, _first_match, _to_number
from utils.text_parsers import *
from utils.table_parsers import *
import re

@traceable(name="aggregate_rent_roll")
def aggregate_rent_roll(dfs: List[pd.DataFrame], fallback_pdf_paths: List[str] = []) -> Dict[str, Any]:
    """
    Compute current_rent_total, market_rent_total and rent_gap_pct from structured dfs.
    If none found, optionally attempt text-based parsing on provided fallback_pdf_paths.
    """
    if dfs:
        total_units = 0
        current_rent_total = 0.0
        market_rent_total = 0.0
        for df in dfs:
            df = _clean_dataframe(df)
            cols = list(df.columns)
            col_rent = _first_match(cols, [r"(^|[^a-z])rent([^a-z]|$)", r"current.*rent", r"base.*rent", r"annual.*rent", r"monthly.*rent"])
            col_market = _first_match(cols, [r"market.*rent", r"asking.*rent"])
            col_psf = _first_match(cols, [r"psf", r"per\s*sf"])
            col_sf = _first_match(cols, [r"sf", r"sq.?ft", r"area"])
            total_units += len(df)
            cur_sum = sum(_to_number(v) or 0.0 for v in df[col_rent].values) if col_rent else 0.0
            mkt_sum = sum(_to_number(v) or 0.0 for v in df[col_market].values) if col_market else 0.0
            if (mkt_sum == 0.0) and col_psf and col_sf:
                monthly = bool(re.search(r"(\/mo|per\s*month|monthly)", col_psf.lower()))
                for _, row in df.iterrows():
                    psf = _to_number(row.get(col_psf))
                    sf = _to_number(row.get(col_sf))
                    if psf and sf:
                        mkt_sum += psf * sf * (12 if monthly else 1)
            current_rent_total += cur_sum
            market_rent_total += mkt_sum
        rent_gap_pct = None
        if market_rent_total > 0:
            rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
        return {
            "total_units": total_units,
            "current_rent_total": current_rent_total if current_rent_total > 0 else None,
            "market_rent_total": market_rent_total if market_rent_total > 0 else None,
            "rent_gap_pct": rent_gap_pct,
        }
    # fallback: try parse from provided PDFs using text heuristics
    for p in fallback_pdf_paths:
        try:
            text_df = parse_rent_roll_from_text(p)
            if not text_df.empty:
                cand = text_df.copy()
                cand["best_amount"] = cand["best_amount"].apply(lambda x: x if x and x > 1000 else None)
                total_units = len(cand)
                current_rent_total = cand["best_amount"].dropna().sum() if "best_amount" in cand else None
                return {
                    "total_units": total_units if total_units>0 else None,
                    "current_rent_total": current_rent_total if current_rent_total and current_rent_total>0 else None,
                    "market_rent_total": None,
                    "rent_gap_pct": None,
                }
        except Exception:
            continue
    return {"total_units": None, "current_rent_total": None, "market_rent_total": None, "rent_gap_pct": None}

@traceable(name="aggregate_t12")
def aggregate_t12(dfs: List[pd.DataFrame], fallback_pdf_paths: List[str] = []) -> Dict[str, Any]:
    if dfs:
        merged = pd.concat([_clean_dataframe(df) for df in dfs], ignore_index=True, sort=False)
        merged.columns = [str(c).strip() for c in merged.columns]
        if _first_match(list(merged.columns), [r"^account", r"^category", r"^description", r"^item"]) is None:
            merged.insert(0, "Item", merged.iloc[:, 0])
        item_col = _first_match(list(merged.columns), [r"item", r"account", r"category", r"description"]) or merged.columns[0]
        num_cols = [c for c in merged.columns if merged[c].map(lambda v: _to_number(v) is not None).sum() > 0]
        candidate_cols = [c for c in num_cols if re.search(r"(ttm|ytd|202|total|current|actual)", str(c).lower())]
        value_col = candidate_cols[-1] if candidate_cols else (num_cols[-1] if num_cols else None)
        def sum_like(patterns: List[str]) -> Optional[float]:
            if not value_col:
                return None
            mask = merged[item_col].astype(str).str.lower().str.contains("|".join(patterns))
            vals = merged.loc[mask, value_col].apply(_to_number).dropna()
            return float(vals.sum()) if not vals.empty else None
        gpr = sum_like([r"gross.*potential.*rent", r"potential.*rent", r"gpr"])
        vacancy = sum_like([r"vacancy", r"credit.*loss", r"loss.*to.*lease"])
        other_income = sum_like([r"other.*income", r"misc.*income", r"parking", r"storage", r"laundry"])
        egi = gpr + (other_income or 0.0) - (vacancy or 0.0) if gpr else None
        opex = sum_like([r"expenses", r"repairs", r"maintenance", r"payroll", r"tax", r"insurance", r"utilities"])
        noi = egi - opex if (egi and opex) else None
        return {
            "gross_potential_rent": gpr,
            "vacancy": vacancy,
            "effective_gross_income": egi,
            "operating_expenses": opex,
            "net_operating_income": noi
        }
    # fallback: attempt text parsing from PDFs
    for p in fallback_pdf_paths:
        t12 = parse_t12_from_text(p)
        if any(v is not None for v in t12.values()):
            return {
                "gross_potential_rent": t12.get("gross_potential_rent"),
                "vacancy": t12.get("vacancy"),
                "effective_gross_income": t12.get("effective_gross_income"),
                "operating_expenses": t12.get("operating_expenses"),
                "net_operating_income": t12.get("net_operating_income")
            }
    return {
        "gross_potential_rent": None,
        "vacancy": None,
        "effective_gross_income": None,
        "operating_expenses": None,
        "net_operating_income": None
    }