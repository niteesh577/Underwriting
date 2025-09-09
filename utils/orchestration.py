
from langsmith import traceable
from file_loaders import *
from table_parsers import *
from helpers import *
from text_parsers import *
from aggregation import *
from metrics import *
from ai_analysis import *
from ai_summary import *
from typing import *
from rag_narrative import *


def run_pipeline(inputs: List[str], overrides: Optional[Dict[str, Any]] = None):
    print("\n--- LOADING FILES ---")
    docs = load_files(inputs)

    print("\n--- EXTRACTING STRUCTURED TABLES ---")
    table_dfs = extract_tables_to_dataframes_from_docs(docs)
    rent_roll_summary = aggregate_rent_roll(table_dfs.get("rent_roll", []), inputs)
    t12_summary = aggregate_t12(table_dfs.get("t12", []), inputs)

    print("\n--- RAG NARRATIVE EXTRACTION ---")
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    narrative_fields = extract_narrative_fields(vs)

    print("\n--- METRICS ---")
    metrics = compute_metrics(t12_summary, rent_roll_summary, narrative_fields, overrides=overrides)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n--- AI UNDERWRITING ANALYSIS ---")
    ai_analysis = generate_underwriting_analysis(narrative_fields, metrics)
    print(json.dumps(ai_analysis, indent=2))

    print("\n--- QUICK SUMMARY ---")
    print(f"Property: {narrative_fields.get('property_name')}")
    print(f"Address: {narrative_fields.get('property_address')}")
    print(f"Year Built: {narrative_fields.get('year_built')}")
    print(f"SqFt: {narrative_fields.get('total_building_sqft')}")
    print(f"NOI (from T12): {t12_summary.get('net_operating_income')}")
    print(f"Expenses (from T12): {t12_summary.get('operating_expenses')}")
    print(f"GPR (from T12): {t12_summary.get('gross_potential_rent')}")
    print(f"Rent Gap %: {metrics.get('rent_gap_pct')}")
    print(f"5-Year IRR: {metrics.get('irr_5yr')}%")
    print(f"Investment Recommendation: {ai_analysis.get('investment_recommendation')}")
    print(f"Key Investment Highlights: {ai_analysis.get('key_investment_highlights')}")
    print(f"Risk Considerations: {ai_analysis.get('risk_considerations')}")
