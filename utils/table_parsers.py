from langsmith import traceable
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from helpers import _clean_dataframe, _guess_is_rent_roll, _guess_is_t12
import pandas as pd
import pdfplumber
import os

@traceable(name="extract_tables_pdfplumber")
def extract_tables_to_dataframes_from_docs(docs: List[Document]) -> Dict[str, List[pd.DataFrame]]:
    """
    Attempt to extract tables from documents that represent PDFs (we will only run pdfplumber on actual pdf paths)
    For Documents made from DataFrames already, use that DataFrame (in metadata) directly.
    """
    out = {"rent_roll": [], "t12": [], "other": []}
    # First, check docs that have dataframe metadata (CSV/Excel)
    for d in docs:
        md = d.metadata or {}
        if "dataframe" in md and isinstance(md["dataframe"], pd.DataFrame):
            df = md["dataframe"]
            df = df.dropna(axis=1, how="all")
            df = df.loc[:, ~(df.columns.astype(str).str.lower().str.contains("^unnamed.*"))]
            df = _clean_dataframe(df)
            if _guess_is_rent_roll(df):
                out["rent_roll"].append(df)
            elif _guess_is_t12(df):
                out["t12"].append(df)
            else:
                out["other"].append(df)

    # For docs that are raw pdf text (from PDF loaders), try to parse tables using pdfplumber directly from the original source if possible.
    # If the Document metadata has a 'source' file path that endswith .pdf, run pdfplumber on that file and extract tables.
    seen_pdf_paths = set()
    for d in docs:
        md = d.metadata or {}
        src = md.get("source")
        if src and str(src).lower().endswith(".pdf") and src not in seen_pdf_paths:
            seen_pdf_paths.add(src)
            try:
                with pdfplumber.open(src) as pdf:
                    for i, page in enumerate(pdf.pages):
                        try:
                            tables = page.extract_tables() or []
                        except Exception:
                            tables = []
                        for tbl in tables:
                            if not tbl or len(tbl) < 2:
                                continue
                            header = [str(h).strip() for h in tbl[0]]
                            rows = [[str(c).strip() for c in r] for r in tbl[1:]]
                            df = pd.DataFrame(rows, columns=header)
                            df = df.dropna(axis=1, how="all")
                            df = df.loc[:, ~(df.columns.astype(str).str.lower().str.contains("^unnamed.*"))]
                            df = _clean_dataframe(df)
                            print(f"\n[PDF {os.path.basename(src)} Page {i+1}] Table Headers: {header}")
                            if _guess_is_rent_roll(df):
                                out["rent_roll"].append(df)
                            elif _guess_is_t12(df):
                                out["t12"].append(df)
                            else:
                                out["other"].append(df)
            except Exception as e:
                print(f"Failed pdfplumber on {src}: {e}")
    return out
