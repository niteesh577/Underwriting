from typing import Dict, Any, Optional

from helpers import _to_number, compute_irr


def compute_metrics(
    t12: Dict[str, Any],
    rent_roll: Dict[str, Any],
    narrative: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute key underwriting metrics from T12, Rent Roll, Narrative, and optional overrides."""
    overrides = overrides or {}

    # Core financials
    noi = _to_number(overrides.get("net_operating_income")) or _to_number(t12.get("net_operating_income"))
    gpr = _to_number(overrides.get("gross_potential_rent")) or _to_number(t12.get("gross_potential_rent"))
    egi = _to_number(overrides.get("effective_gross_income")) or _to_number(t12.get("effective_gross_income"))
    opex = _to_number(overrides.get("operating_expenses")) or _to_number(t12.get("operating_expenses"))

    current_rent_total = _to_number(overrides.get("current_rent_total")) or _to_number(rent_roll.get("current_rent_total"))
    market_rent_total = _to_number(overrides.get("market_rent_total")) or _to_number(rent_roll.get("market_rent_total"))

    rent_gap_pct = overrides.get("rent_gap_pct") or rent_roll.get("rent_gap_pct")
    if rent_gap_pct is None and current_rent_total and market_rent_total:
        rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0

    sqft = _to_number(overrides.get("total_building_sqft")) or _to_number(narrative.get("total_building_sqft"))
    units = _to_number(overrides.get("total_units")) or _to_number(narrative.get("total_units_or_suites"))

    purchase_price = _to_number(overrides.get("purchase_price"))
    if not purchase_price and noi:
        purchase_price = noi / 0.075  # assume 7.5% cap if price not given

    annual_debt_service = _to_number(overrides.get("annual_debt_service")) or (purchase_price * 0.05 if purchase_price else None)
    equity_invested = _to_number(overrides.get("equity_invested")) or (purchase_price * 0.25 if purchase_price else None)

    # Metrics
    cap_rate = (noi / purchase_price) if (noi and purchase_price) else None
    dscr = (noi / annual_debt_service) if (noi and annual_debt_service) else None
    coc = ((noi - annual_debt_service) / equity_invested) if (noi and annual_debt_service and equity_invested) else None
    price_per_sf = (purchase_price / sqft) if (purchase_price and sqft) else None
    price_per_unit = (purchase_price / units) if (purchase_price and units) else None
    break_even_occ = ((opex or 0.0) + (annual_debt_service or 0.0)) / egi if egi else None

    # Compute 5-year IRR using NOI - debt service as proxy cash flow
    if noi is not None and annual_debt_service is not None:
        base_cash_flow = noi - annual_debt_service
        cash_flows = [base_cash_flow * ((1 + 0.02) ** i) for i in range(1, 6)]
        cash_flows.insert(0, -equity_invested if equity_invested else -purchase_price)
        try:
            irr_5yr = compute_irr(cash_flows)
        except Exception:
            irr_5yr = None
    else:
        irr_5yr = None

    return {
        "cap_rate": cap_rate,
        "dscr": dscr,
        "coc_return": coc,
        "irr_5yr": irr_5yr,
        "rent_gap_pct": rent_gap_pct,
        "price_per_sqft": price_per_sf,
        "price_per_unit": price_per_unit,
        "break_even_occupancy": break_even_occ,
    }
