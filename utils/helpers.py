# utils/helpers.py
# -*- coding: utf-8 -*-
import re
import math
import pandas as pd
import numpy as np
from typing import List, Optional

def _to_number(x) -> Optional[float]:
    """Parse common currency/number formats safely -> float."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "-", "—", "–", "N/A", "NA", "None"):
        return None
    try:
        return float(s)
    except Exception:
        s = s.replace("(", "-").replace(")", "")
        try:
            return float(s)
        except Exception:
            return None

def _first_match(cols: List[str], patterns: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for p in patterns:
        for lc, orig in lower.items():
            if re.search(p, lc):
                return orig
    return None

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique, clean column headers for parsing."""
    df = df.copy()
    cols = []
    seen = {}
    for c in df.columns:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    return df

def _guess_is_rent_roll(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    wanted = [
        "tenant", "suite", "unit", "lease", "psf", "rent",
        "term", "start", "end", "sf", "sqft", "occupied", "vacant"
    ]
    score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
    return score >= 3

def _guess_is_t12(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    wanted = [
        "income", "revenue", "rent", "gpr", "noi", "expenses",
        "egi", "vacancy", "operating", "total"
    ]
    score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
    return score >= 2

def compute_irr(cash_flows: List[float], guess: float = 0.1) -> Optional[float]:
    """Compute IRR using numpy."""
    if not cash_flows:
        return None
    try:
        return np.irr(cash_flows) * 100  # in percentage
    except:
        return None
