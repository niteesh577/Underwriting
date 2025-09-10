# # -*- coding: utf-8 -*-
# # Hybrid CRE Underwriting Pipeline: Structured Tables + RAG + Metrics
# # pip install -U pandas numpy langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith pdfplumber

# import os
# import re
# import math
# from typing import List, Dict, Any, Optional

# # Avoid macOS OpenMP crash (FAISS/numpy)
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

# import pandas as pd
# import numpy as np
# import pdfplumber
# from dotenv import load_dotenv

# from langsmith import traceable
# from langchain_community.document_loaders import (
#     PDFPlumberLoader,
#     PyPDFLoader,
#     UnstructuredPDFLoader,
#     PyPDFium2Loader,
#     PyMuPDFLoader,
#     PDFMinerLoader,
#     PDFMinerPDFasHTMLLoader,
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document

# # ------------- CONFIG -------------
# load_dotenv()
# PDF_PATH = "ParkCenter_400_OM.pdf"
# EMBED_MODEL = "text-embedding-3-small"
# LLM_MODEL = "gpt-4o-mini"
# TABLE_CHUNK_SIZE = 1000
# TABLE_CHUNK_OVERLAP = 150

# # ------------- HELPERS -------------

# def _to_number(x: Any) -> Optional[float]:
#     """Parse common currency/number formats safely -> float."""
#     if x is None:
#         return None
#     if isinstance(x, (int, float, np.number)):
#         return float(x)
#     s = str(x)
#     s = s.replace("$", "").replace(",", "").replace("%", "").strip()
#     if s in ("", "-", "—", "–", "N/A", "NA"):
#         return None
#     try:
#         return float(s)
#     except Exception:
#         s = s.replace("(", "-").replace(")", "")
#         try:
#             return float(s)
#         except Exception:
#             return None

# def _first_match(cols: List[str], patterns: List[str]) -> Optional[str]:
#     lower = {c.lower(): c for c in cols}
#     for p in patterns:
#         for lc, orig in lower.items():
#             if re.search(p, lc):
#                 return orig
#     return None

# def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure unique, clean column headers for parsing."""
#     df = df.copy()
#     cols = []
#     seen = {}
#     for c in df.columns:
#         c = str(c).strip()
#         if c in seen:
#             seen[c] += 1
#             c = f"{c}_{seen[c]}"
#         else:
#             seen[c] = 0
#         cols.append(c)
#     df.columns = cols
#     return df

# def _guess_is_rent_roll(df: pd.DataFrame) -> bool:
#     cols = [str(c) for c in df.columns]
#     wanted = [
#         "tenant", "suite", "unit", "lease", "psf", "rent",
#         "term", "start", "end", "sf", "sqft", "occupied", "vacant"
#     ]
#     score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
#     return score >= 3

# def _guess_is_t12(df: pd.DataFrame) -> bool:
#     cols = [str(c) for c in df.columns]
#     wanted = [
#         "income", "revenue", "rent", "gpr", "noi", "expenses",
#         "egi", "vacancy", "operating", "total"
#     ]
#     score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
#     return score >= 2

# # ------------- LOADERS -------------

# @traceable(name="load_pdf_multi")
# def load_pdf_multi(path: str) -> List[Document]:
#     docs: List[Document] = []
#     loaders = [
#         ("PDFPlumberLoader", PDFPlumberLoader(path)),
#         ("PyPDFLoader", PyPDFLoader(path)),
#         ("UnstructuredPDFLoader (OCR)", UnstructuredPDFLoader(path, strategy="ocr_only")),
#         ("PyPDFium2Loader", PyPDFium2Loader(path)),
#         ("PyMuPDFLoader", PyMuPDFLoader(path)),
#         ("PDFMinerLoader", PDFMinerLoader(path)),
#         ("PDFMinerPDFasHTMLLoader", PDFMinerPDFasHTMLLoader(path)),
#     ]
#     for name, loader in loaders:
#         try:
#             new_docs = loader.load()
#             docs.extend(new_docs)
#             print(f"{name} extracted {len(new_docs)} docs")
#         except Exception as e:
#             print(f"{name} failed: {e}")
#     seen = set()
#     unique_docs: List[Document] = []
#     for d in docs:
#         if d.page_content not in seen:
#             unique_docs.append(d)
#             seen.add(d.page_content)
#     print(f"Total unique docs after merging: {len(unique_docs)}")
#     return unique_docs

# # ------------- TABLE PARSING -------------

# @traceable(name="extract_tables_pdfplumber")
# def extract_tables_to_dataframes(path: str) -> Dict[str, List[pd.DataFrame]]:
#     out = {"rent_roll": [], "t12": [], "other": []}
#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             try:
#                 tables = page.extract_tables() or []
#             except Exception:
#                 tables = []
#             for tbl in tables:
#                 if not tbl or len(tbl) < 2:
#                     continue
#                 header = [str(h).strip() for h in tbl[0]]
#                 rows = [[str(c).strip() for c in r] for r in tbl[1:]]
#                 df = pd.DataFrame(rows, columns=header)
#                 df = df.dropna(axis=1, how="all")
#                 df = df.loc[:, ~(df.columns.astype(str).str.lower().str.contains("^unnamed.*"))]
#                 df = _clean_dataframe(df)

#                 print(f"\n[Page {i+1}] Table Headers: {header}")  # Debugging

#                 if _guess_is_rent_roll(df):
#                     out["rent_roll"].append(df)
#                 elif _guess_is_t12(df):
#                     out["t12"].append(df)
#                 else:
#                     out["other"].append(df)
#     return out

# # ------------- AGGREGATION -------------

# @traceable(name="aggregate_rent_roll")
# def aggregate_rent_roll(dfs: List[pd.DataFrame]) -> Dict[str, Any]:
#     if not dfs:
#         return {"total_units": None, "current_rent_total": None, "market_rent_total": None, "rent_gap_pct": None}
#     total_units = 0
#     current_rent_total = 0.0
#     market_rent_total = 0.0
#     for df in dfs:
#         cols = list(df.columns)
#         col_rent = _first_match(cols, [r"(^|[^a-z])rent([^a-z]|$)", r"current.*rent", r"base.*rent", r"annual.*rent"])
#         col_market = _first_match(cols, [r"market.*rent", r"asking.*rent"])
#         col_psf = _first_match(cols, [r"psf", r"per\s*sf"])
#         col_sf = _first_match(cols, [r"sf", r"sq.?ft", r"area"])
#         total_units += len(df)
#         cur_sum = sum(_to_number(v) or 0.0 for v in df[col_rent].values) if col_rent else 0.0
#         mkt_sum = sum(_to_number(v) or 0.0 for v in df[col_market].values) if col_market else 0.0
#         if (mkt_sum == 0.0) and col_psf and col_sf:
#             monthly = bool(re.search(r"(\/mo|per\s*month|monthly)", col_psf.lower()))
#             for _, row in df.iterrows():
#                 psf = _to_number(row.get(col_psf))
#                 sf = _to_number(row.get(col_sf))
#                 if psf and sf:
#                     mkt_sum += psf * sf * (12 if monthly else 1)
#         current_rent_total += cur_sum
#         market_rent_total += mkt_sum
#     rent_gap_pct = None
#     if market_rent_total > 0:
#         rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
#     return {
#         "total_units": total_units,
#         "current_rent_total": current_rent_total if current_rent_total > 0 else None,
#         "market_rent_total": market_rent_total if market_rent_total > 0 else None,
#         "rent_gap_pct": rent_gap_pct,
#     }

# @traceable(name="aggregate_t12")
# def aggregate_t12(dfs: List[pd.DataFrame]) -> Dict[str, Any]:
#     if not dfs:
#         return {"gross_potential_rent": None, "vacancy": None, "effective_gross_income": None,
#                 "operating_expenses": None, "net_operating_income": None}
#     merged = pd.concat([_clean_dataframe(df) for df in dfs], ignore_index=True, sort=False)
#     merged.columns = [str(c).strip() for c in merged.columns]
#     if _first_match(list(merged.columns), [r"^account", r"^category", r"^description", r"^item"]) is None:
#         merged.insert(0, "Item", merged.iloc[:, 0])
#     item_col = _first_match(list(merged.columns), [r"item", r"account", r"category", r"description"]) or merged.columns[0]
#     num_cols = [c for c in merged.columns if merged[c].map(lambda v: _to_number(v) is not None).sum() > 0]
#     candidate_cols = [c for c in num_cols if re.search(r"(ttm|ytd|202|total|current|actual)", str(c).lower())]
#     value_col = candidate_cols[-1] if candidate_cols else (num_cols[-1] if num_cols else None)

#     def sum_like(patterns: List[str]) -> Optional[float]:
#         if not value_col:
#             return None
#         mask = merged[item_col].astype(str).str.lower().str.contains("|".join(patterns))
#         vals = merged.loc[mask, value_col].apply(_to_number).dropna()
#         return float(vals.sum()) if not vals.empty else None

#     gpr = sum_like([r"gross.*potential.*rent", r"potential.*rent", r"gpr"])
#     vacancy = sum_like([r"vacancy", r"credit.*loss", r"loss.*to.*lease"])
#     other_income = sum_like([r"other.*income", r"misc.*income", r"parking", r"storage", r"laundry"])
#     egi = gpr + (other_income or 0.0) - (vacancy or 0.0) if gpr else None
#     opex = sum_like([r"expenses", r"repairs", r"maintenance", r"payroll", r"tax", r"insurance", r"utilities"])
#     noi = egi - opex if (egi and opex) else None
#     return {
#         "gross_potential_rent": gpr,
#         "vacancy": vacancy,
#         "effective_gross_income": egi,
#         "operating_expenses": opex,
#         "net_operating_income": noi
#     }

# # ------------- RAG -------------

# def _format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# @traceable(name="extract_narrative_fields")
# def extract_narrative_fields(vs) -> Dict[str, Any]:
#     retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Extract ONLY from the provided context. If not found, write 'I don't know'. Respond in strict JSON."),
#         ("human", """Extract these fields:
# - property_name
# - property_address
# - property_type
# - year_built
# - renovation_year
# - number_of_stories
# - total_units_or_suites
# - total_building_sqft
# - amenities

# Context:
# {context}""")
#     ])
#     parallel = RunnableParallel({
#         "context": retriever | RunnableLambda(_format_docs),
#         "question": RunnablePassthrough()
#     })
#     chain = parallel | prompt | llm | StrOutputParser()
#     query = "Extract property details (name, address, type, year built, sqft, units, amenities)"
#     out = chain.invoke(query)

#     # Strip code fences if present
#     out = re.sub(r"^```json|```$", "", out.strip(), flags=re.MULTILINE).strip()

#     try:
#         import json
#         return json.loads(out)
#     except Exception:
#         return {"raw": out}

# # ------------- METRICS -------------

# def compute_metrics(inputs: Dict[str, Any]) -> Dict[str, Any]:
#     purchase_price = _to_number(inputs.get("purchase_price"))
#     noi = _to_number(inputs.get("net_operating_income"))
#     annual_debt_service = _to_number(inputs.get("annual_debt_service"))
#     cfb4tax = _to_number(inputs.get("cash_flow_before_taxes"))
#     equity_invested = _to_number(inputs.get("equity_invested"))
#     total_building_sqft = _to_number(inputs.get("total_building_sqft"))
#     total_units = _to_number(inputs.get("total_units"))
#     gpr = _to_number(inputs.get("gross_potential_rent"))
#     opex = _to_number(inputs.get("operating_expenses"))
#     current_rent_total = _to_number(inputs.get("current_rent_total"))
#     market_rent_total = _to_number(inputs.get("market_rent_total"))
#     rent_gap_pct = inputs.get("rent_gap_pct")
#     if rent_gap_pct is None and current_rent_total and market_rent_total:
#         rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
#     cap_rate = (noi / purchase_price) if (noi and purchase_price) else None
#     dscr = (noi / annual_debt_service) if (noi and annual_debt_service) else None
#     coc = (cfb4tax / equity_invested) if (cfb4tax and equity_invested) else None
#     price_per_sf = (purchase_price / total_building_sqft) if (purchase_price and total_building_sqft) else None
#     price_per_unit = (purchase_price / total_units) if (purchase_price and total_units) else None
#     break_even_occ = ((opex or 0.0) + (annual_debt_service or 0.0)) / gpr if gpr else None
#     return {
#         "cap_rate": cap_rate,
#         "dscr": dscr,
#         "coc_return": coc,
#         "irr_5yr": None,
#         "rent_gap_pct": rent_gap_pct,
#         "price_per_sqft": price_per_sf,
#         "price_per_unit": price_per_unit,
#         "break_even_occupancy": break_even_occ
#     }

# # ------------- MAIN -------------

# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=TABLE_CHUNK_SIZE, chunk_overlap=TABLE_CHUNK_OVERLAP):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     final_splits = []
#     for doc in docs:
#         text = doc.page_content
#         if re.search(r"\d+\s+\d+", text) and ("\n" in text or "\t" in text):
#             final_splits.append(doc)
#         else:
#             final_splits.extend(splitter.split_documents([doc]))
#     print(f"Total chunks after table-aware splitting: {len(final_splits)}")
#     return final_splits

# @traceable(name="setup_rag")
# def setup_rag(pdf_path: str):
#     docs = load_pdf_multi(pdf_path)
#     splits = split_documents(docs)
#     return build_vectorstore(splits)

# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = OpenAIEmbeddings(model=EMBED_MODEL)
#     return FAISS.from_documents(splits, emb)

# def main(pdf_path: str = PDF_PATH):
#     print("\n--- STRUCTURED TABLE PARSING ---")
#     tables = extract_tables_to_dataframes(pdf_path)
#     rent_roll_summary = aggregate_rent_roll(tables["rent_roll"])
#     t12_summary = aggregate_t12(tables["t12"])
#     print("\nRent Roll Summary:", rent_roll_summary)
#     print("T12 Summary:", t12_summary)

#     print("\n--- RAG NARRATIVE EXTRACTION ---")
#     vs = setup_rag(pdf_path)
#     narrative = extract_narrative_fields(vs)
#     print("Narrative Fields:", narrative)

#     inputs = {
#         "net_operating_income": t12_summary.get("net_operating_income"),
#         "operating_expenses": t12_summary.get("operating_expenses"),
#         "gross_potential_rent": t12_summary.get("gross_potential_rent"),
#         "current_rent_total": rent_roll_summary.get("current_rent_total"),
#         "market_rent_total": rent_roll_summary.get("market_rent_total"),
#         "rent_gap_pct": rent_roll_summary.get("rent_gap_pct"),
#         "total_units": rent_roll_summary.get("total_units"),
#         "total_building_sqft": narrative.get("total_building_sqft"),
#         "purchase_price": None,
#         "annual_debt_service": None,
#         "cash_flow_before_taxes": None,
#         "equity_invested": None,
#     }

#     print("\n--- METRICS ---")
#     metrics = compute_metrics(inputs)
#     for k, v in metrics.items():
#         if isinstance(v, float):
#             if "rate" in k or "pct" in k or "occupancy" in k:
#                 print(f"{k}: {v:.2%}")
#             elif "price" in k:
#                 print(f"{k}: ${v:,.2f}")
#             else:
#                 print(f"{k}: {v:,.2f}")
#         else:
#             print(f"{k}: {v}")

#     print("\n--- QUICK SUMMARY ---")
#     print(f"Property: {narrative.get('property_name', 'Unknown')}")
#     print(f"Address: {narrative.get('property_address', 'Unknown')}")
#     print(f"Year Built: {narrative.get('year_built', 'Unknown')}")
#     print(f"SqFt: {narrative.get('total_building_sqft', 'Unknown')}")
#     print(f"NOI (from T12): {t12_summary.get('net_operating_income')}")
#     print(f"Expenses (from T12): {t12_summary.get('operating_expenses')}")
#     print(f"GPR (from T12): {t12_summary.get('gross_potential_rent')}")
#     print(f"Rent Gap % (from Rent Roll): {metrics.get('rent_gap_pct'):.2f}%" if metrics.get("rent_gap_pct") else "Rent Gap %: Unknown")

# if __name__ == "__main__":
#     main(PDF_PATH)





# # -*- coding: utf-8 -*-
# # Hybrid CRE Underwriting Pipeline: Structured Tables + RAG + Metrics
# # Requires:
# # pip install -U pandas numpy langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith pdfplumber

# import os
# import re
# import math
# from typing import List, Dict, Any, Optional
# import csv

# # Avoid macOS OpenMP crash (FAISS/numpy)
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

# import pandas as pd
# import numpy as np
# import pdfplumber
# from dotenv import load_dotenv

# from langsmith import traceable
# from langchain_community.document_loaders import (
#     PDFPlumberLoader,
#     PyPDFLoader,
#     UnstructuredPDFLoader,
#     PyPDFium2Loader,
#     PyMuPDFLoader,
#     PDFMinerLoader,
#     PDFMinerPDFasHTMLLoader,
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document

# # ------------- CONFIG -------------
# load_dotenv()
# PDF_PATH = "ParkCenter_400_Underwriting_Assumptions.pdf"
# EMBED_MODEL = "text-embedding-3-small"
# LLM_MODEL = "gpt-4o-mini"
# TABLE_CHUNK_SIZE = 1000
# TABLE_CHUNK_OVERLAP = 150

# # ------------- HELPERS -------------

# def _to_number(x: Any) -> Optional[float]:
#     """Parse common currency/number formats safely -> float."""
#     if x is None:
#         return None
#     if isinstance(x, (int, float, np.number)):
#         return float(x)
#     s = str(x)
#     s = s.replace("$", "").replace(",", "").replace("%", "").strip()
#     if s in ("", "-", "—", "–", "N/A", "NA", "None"):
#         return None
#     try:
#         return float(s)
#     except Exception:
#         s = s.replace("(", "-").replace(")", "")
#         try:
#             return float(s)
#         except Exception:
#             return None

# def _first_match(cols: List[str], patterns: List[str]) -> Optional[str]:
#     lower = {c.lower(): c for c in cols}
#     for p in patterns:
#         for lc, orig in lower.items():
#             if re.search(p, lc):
#                 return orig
#     return None

# def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure unique, clean column headers for parsing."""
#     df = df.copy()
#     cols = []
#     seen = {}
#     for c in df.columns:
#         c = str(c).strip()
#         if c in seen:
#             seen[c] += 1
#             c = f"{c}_{seen[c]}"
#         else:
#             seen[c] = 0
#         cols.append(c)
#     df.columns = cols
#     return df

# def _guess_is_rent_roll(df: pd.DataFrame) -> bool:
#     cols = [str(c) for c in df.columns]
#     wanted = [
#         "tenant", "suite", "unit", "lease", "psf", "rent",
#         "term", "start", "end", "sf", "sqft", "occupied", "vacant"
#     ]
#     score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
#     return score >= 3

# def _guess_is_t12(df: pd.DataFrame) -> bool:
#     cols = [str(c) for c in df.columns]
#     wanted = [
#         "income", "revenue", "rent", "gpr", "noi", "expenses",
#         "egi", "vacancy", "operating", "total"
#     ]
#     score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
#     return score >= 2

# # ------------- LOADERS -------------

# @traceable(name="load_pdf_multi")
# def load_pdf_multi(path: str) -> List[Document]:
#     docs: List[Document] = []
#     loaders = [
#         ("PDFPlumberLoader", PDFPlumberLoader(path)),
#         ("PyPDFLoader", PyPDFLoader(path)),
#         ("UnstructuredPDFLoader (OCR)", UnstructuredPDFLoader(path, strategy="ocr_only")),
#         ("PyPDFium2Loader", PyPDFium2Loader(path)),
#         ("PyMuPDFLoader", PyMuPDFLoader(path)),
#         ("PDFMinerLoader", PDFMinerLoader(path)),
#         ("PDFMinerPDFasHTMLLoader", PDFMinerPDFasHTMLLoader(path)),
#     ]
#     for name, loader in loaders:
#         try:
#             new_docs = loader.load()
#             docs.extend(new_docs)
#             print(f"{name} extracted {len(new_docs)} docs")
#         except Exception as e:
#             print(f"{name} failed: {e}")
#     seen = set()
#     unique_docs: List[Document] = []
#     for d in docs:
#         if d.page_content not in seen:
#             unique_docs.append(d)
#             seen.add(d.page_content)
#     print(f"Total unique docs after merging: {len(unique_docs)}")
#     return unique_docs

# # ------------- TABLE PARSING (pdfplumber) -------------

# @traceable(name="extract_tables_pdfplumber")
# def extract_tables_to_dataframes(path: str) -> Dict[str, List[pd.DataFrame]]:
#     out = {"rent_roll": [], "t12": [], "other": []}
#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             try:
#                 tables = page.extract_tables() or []
#             except Exception:
#                 tables = []
#             for tbl in tables:
#                 if not tbl or len(tbl) < 2:
#                     continue
#                 header = [str(h).strip() for h in tbl[0]]
#                 rows = [[str(c).strip() for c in r] for r in tbl[1:]]
#                 df = pd.DataFrame(rows, columns=header)
#                 df = df.dropna(axis=1, how="all")
#                 df = df.loc[:, ~(df.columns.astype(str).str.lower().str.contains("^unnamed.*"))]
#                 df = _clean_dataframe(df)

#                 # Debug header
#                 print(f"\n[Page {i+1}] Table Headers: {header}")

#                 if _guess_is_rent_roll(df):
#                     out["rent_roll"].append(df)
#                 elif _guess_is_t12(df):
#                     out["t12"].append(df)
#                 else:
#                     out["other"].append(df)
#     return out

# # ------------- TEXT-BASED FALLBACK PARSERS -------------

# def _extract_amounts_from_line(line: str) -> List[float]:
#     """Return list of monetary numbers found in a line."""
#     amounts = []
#     for m in re.finditer(r"(-?\$?\d{1,3}(?:[,\d]{0,}|(?:\d+))(?:\.\d{1,2})?)", line):
#         val = m.group(0)
#         if '$' in val or re.search(r"\d", val):
#             num = _to_number(val)
#             if num is not None:
#                 amounts.append(num)
#     return amounts

# @traceable(name="parse_rent_roll_from_text")
# def parse_rent_roll_from_text(path: str) -> pd.DataFrame:
#     """
#     Heuristic parser: scan page text for suites/tenant lines and nearby numbers.
#     Returns a dataframe with candidate rows and parsed fields (best-effort).
#     """
#     candidates = []
#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text() or ""
#             # Normalize spacing: many OMs have newlines between suite/tenant/psf
#             lines = [l.strip() for l in text.splitlines() if l.strip()]
#             # Join small groups of nearby lines to make candidate records
#             for idx, line in enumerate(lines):
#                 # detect suite number or "Suite" or a line that looks like "<number> <Tenant>"
#                 if re.search(r"(^suite\s*\d+)|(^\d{2,4}\b)", line.lower()) or re.search(r"\b(suite|ste|#)\b", line.lower()):
#                     window = " | ".join(lines[max(0, idx-2): idx+4])  # context window
#                     amounts = _extract_amounts_from_line(window)
#                     sqft = None
#                     # try extract sqft (patterns like 10,143 or 10143)
#                     m_sq = re.search(r"(\d{3,6})(?:\s?(?:sf|sq\.?ft|s\.f\.))", window.replace(",", "").lower())
#                     if m_sq:
#                         sqft = _to_number(m_sq.group(1))
#                     # try extract lease end or date
#                     lease = None
#                     m_lease = re.search(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\w\.-]*\s*\-?\s*\d{4})", window, flags=re.I)
#                     if m_lease:
#                         lease = m_lease.group(0)
#                     # tenant extraction (first capitalized token sequence)
#                     m_tenant = re.search(r"\b(?:suite|ste|#)?\s*\d{0,4}\s*[-:]*\s*([A-Z][A-Za-z0-9&,\.\- ]{2,80})", window)
#                     tenant = m_tenant.group(1).strip() if m_tenant else None
#                     candidates.append({
#                         "page": i+1,
#                         "window": window,
#                         "tenant": tenant,
#                         "sqft": sqft,
#                         "amounts": amounts,
#                         "lease_hint": lease
#                     })
#     # Make DataFrame
#     df = pd.DataFrame(candidates)
#     # Postprocess: attempt to choose best amount as "current rent" or "psf" heuristically:
#     def pick_amount(row):
#         if not row["amounts"]:
#             return None
#         # prefer amounts < 100 (likely psf) or amounts that look like annual rent (>1000)
#         nums = sorted(row["amounts"])
#         # prefer largest as annual rent
#         return nums[-1]
#     if not df.empty:
#         df["best_amount"] = df.apply(pick_amount, axis=1)
#     # Save debug CSV
#     df.to_csv("debug_rent_roll_candidates.csv", index=False)
#     return df

# @traceable(name="parse_t12_from_text")
# def parse_t12_from_text(path: str) -> Dict[str, Optional[float]]:
#     """
#     Heuristic parser: search for line items in page text that match common T12 line descriptions
#     and capture nearest numeric amounts.
#     Returns a dict with gpr, vacancy, egi, opex, noi (best-effort).
#     """
#     text_all = ""
#     with pdfplumber.open(path) as pdf:
#         for page in pdf.pages:
#             text_all += "\n" + (page.extract_text() or "")

#     # normalize
#     txt = re.sub(r"\s{2,}", " ", text_all)
#     lines = [l.strip() for l in txt.splitlines() if l.strip()]
#     # Collect candidate matches: lines with keywords + numbers
#     candidates = []
#     for ln in lines:
#         if re.search(r"(gross.*potential.*rent|potential.*rent|gpr|effective gross|effective gross income|net operating income|operating expenses|total expenses|vacancy|credit loss|other income|misc income|parking|laundry)", ln, flags=re.I):
#             nums = _extract_amounts_from_line(ln)
#             candidates.append({"line": ln, "amounts": nums})
#     # Write debug CSV
#     with open("debug_t12_candidates.csv", "w", newline="", encoding="utf-8") as fh:
#         writer = csv.DictWriter(fh, fieldnames=["line", "amounts"])
#         writer.writeheader()
#         for c in candidates:
#             writer.writerow({"line": c["line"], "amounts": ";".join(str(a) for a in c["amounts"])})
#     # Heuristic picks:
#     def pick_for_pattern(pats):
#         for c in candidates:
#             for p in pats:
#                 if re.search(p, c["line"], flags=re.I):
#                     if c["amounts"]:
#                         return c["amounts"][-1]  # last numeric token
#         return None
#     gpr = pick_for_pattern([r"gross.*potential.*rent", r"potential.*rent", r"gpr"])
#     vacancy = pick_for_pattern([r"vacancy", r"credit.*loss", r"loss.*to.*lease"])
#     other_income = pick_for_pattern([r"other.*income", r"misc.*income", r"parking", r"laundry", r"storage"])
#     egi_candidate = pick_for_pattern([r"effective gross income", r"effective.*gross", r"egi"])
#     opex = pick_for_pattern([r"operating expenses", r"total expenses", r"expenses", r"total operating expenses"])
#     noi = pick_for_pattern([r"net operating income", r"noi"])
#     # If EGI not found, compute from GPR +/- others
#     egi = egi_candidate
#     if egi is None and gpr is not None:
#         egi = gpr - (vacancy or 0.0) + (other_income or 0.0)
#     return {
#         "gross_potential_rent": gpr,
#         "vacancy": vacancy,
#         "other_income": other_income,
#         "effective_gross_income": egi,
#         "operating_expenses": opex,
#         "net_operating_income": noi
#     }

# # ------------- AGGREGATION FROM STRUCTURED TABLES -------------

# @traceable(name="aggregate_rent_roll")
# def aggregate_rent_roll(dfs: List[pd.DataFrame]) -> Dict[str, Any]:
#     """
#     Try to compute current_rent_total, market_rent_total and rent_gap_pct from structured dfs.
#     If none found, fallback to text-based parser.
#     """
#     # If structured dfs present, attempt to extract
#     if dfs:
#         total_units = 0
#         current_rent_total = 0.0
#         market_rent_total = 0.0
#         for df in dfs:
#             df = _clean_dataframe(df)
#             cols = list(df.columns)
#             col_rent = _first_match(cols, [r"(^|[^a-z])rent([^a-z]|$)", r"current.*rent", r"base.*rent", r"annual.*rent", r"monthly.*rent"])
#             col_market = _first_match(cols, [r"market.*rent", r"asking.*rent"])
#             col_psf = _first_match(cols, [r"psf", r"per\s*sf"])
#             col_sf = _first_match(cols, [r"sf", r"sq.?ft", r"area"])
#             total_units += len(df)
#             cur_sum = sum(_to_number(v) or 0.0 for v in df[col_rent].values) if col_rent else 0.0
#             mkt_sum = sum(_to_number(v) or 0.0 for v in df[col_market].values) if col_market else 0.0
#             if (mkt_sum == 0.0) and col_psf and col_sf:
#                 monthly = bool(re.search(r"(\/mo|per\s*month|monthly)", col_psf.lower()))
#                 for _, row in df.iterrows():
#                     psf = _to_number(row.get(col_psf))
#                     sf = _to_number(row.get(col_sf))
#                     if psf and sf:
#                         mkt_sum += psf * sf * (12 if monthly else 1)
#             current_rent_total += cur_sum
#             market_rent_total += mkt_sum
#         rent_gap_pct = None
#         if market_rent_total > 0:
#             rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
#         return {
#             "total_units": total_units,
#             "current_rent_total": current_rent_total if current_rent_total > 0 else None,
#             "market_rent_total": market_rent_total if market_rent_total > 0 else None,
#             "rent_gap_pct": rent_gap_pct,
#         }
#     # Else fallback to text parser
#     text_df = parse_rent_roll_from_text(PDF_PATH)
#     # Heuristic aggregation on text_df
#     if text_df.empty:
#         return {"total_units": None, "current_rent_total": None, "market_rent_total": None, "rent_gap_pct": None}
#     # Assume best_amount is annual rent if >1000 else PSF fallback (very rough)
#     cand = text_df.copy()
#     cand["best_amount"] = cand["best_amount"].apply(lambda x: x if x and x > 1000 else None)
#     total_units = len(cand)
#     current_rent_total = cand["best_amount"].dropna().sum() if "best_amount" in cand else None
#     return {
#         "total_units": total_units if total_units>0 else None,
#         "current_rent_total": current_rent_total if current_rent_total and current_rent_total>0 else None,
#         "market_rent_total": None,
#         "rent_gap_pct": None,
#     }

# @traceable(name="aggregate_t12")
# def aggregate_t12(dfs: List[pd.DataFrame]) -> Dict[str, Any]:
#     # If structured dfs present try to compute
#     if dfs:
#         merged = pd.concat([_clean_dataframe(df) for df in dfs], ignore_index=True, sort=False)
#         merged.columns = [str(c).strip() for c in merged.columns]
#         if _first_match(list(merged.columns), [r"^account", r"^category", r"^description", r"^item"]) is None:
#             merged.insert(0, "Item", merged.iloc[:, 0])
#         item_col = _first_match(list(merged.columns), [r"item", r"account", r"category", r"description"]) or merged.columns[0]
#         num_cols = [c for c in merged.columns if merged[c].map(lambda v: _to_number(v) is not None).sum() > 0]
#         candidate_cols = [c for c in num_cols if re.search(r"(ttm|ytd|202|total|current|actual)", str(c).lower())]
#         value_col = candidate_cols[-1] if candidate_cols else (num_cols[-1] if num_cols else None)
#         def sum_like(patterns: List[str]) -> Optional[float]:
#             if not value_col:
#                 return None
#             mask = merged[item_col].astype(str).str.lower().str.contains("|".join(patterns))
#             vals = merged.loc[mask, value_col].apply(_to_number).dropna()
#             return float(vals.sum()) if not vals.empty else None
#         gpr = sum_like([r"gross.*potential.*rent", r"potential.*rent", r"gpr"])
#         vacancy = sum_like([r"vacancy", r"credit.*loss", r"loss.*to.*lease"])
#         other_income = sum_like([r"other.*income", r"misc.*income", r"parking", r"storage", r"laundry"])
#         egi = gpr + (other_income or 0.0) - (vacancy or 0.0) if gpr else None
#         opex = sum_like([r"expenses", r"repairs", r"maintenance", r"payroll", r"tax", r"insurance", r"utilities"])
#         noi = egi - opex if (egi and opex) else None
#         return {
#             "gross_potential_rent": gpr,
#             "vacancy": vacancy,
#             "effective_gross_income": egi,
#             "operating_expenses": opex,
#             "net_operating_income": noi
#         }
#     # Fallback text parse
#     t12 = parse_t12_from_text(PDF_PATH)
#     return {
#         "gross_potential_rent": t12.get("gross_potential_rent"),
#         "vacancy": t12.get("vacancy"),
#         "effective_gross_income": t12.get("effective_gross_income"),
#         "operating_expenses": t12.get("operating_expenses"),
#         "net_operating_income": t12.get("net_operating_income")
#     }

# # ------------- RAG & Narrative -------------

# def _format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=TABLE_CHUNK_SIZE, chunk_overlap=TABLE_CHUNK_OVERLAP):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     final_splits = []
#     for doc in docs:
#         text = doc.page_content
#         if re.search(r"\d+\s+\d+", text) and ("\n" in text or "\t" in text):
#             final_splits.append(doc)
#         else:
#             final_splits.extend(splitter.split_documents([doc]))
#     print(f"Total chunks after table-aware splitting: {len(final_splits)}")
#     return final_splits

# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = OpenAIEmbeddings(model=EMBED_MODEL)
#     return FAISS.from_documents(splits, emb)

# @traceable(name="extract_narrative_fields")
# def extract_narrative_fields(vs) -> Dict[str, Any]:
#     retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Extract ONLY from the provided context. If not found, write 'I don't know'. Respond in strict JSON."),
#         ("human", """Extract these fields:
# - property_name
# - property_address
# - property_type
# - year_built
# - renovation_year
# - number_of_stories
# - total_units_or_suites
# - total_building_sqft
# - amenities

# Context:
# {context}""")
#     ])
#     parallel = RunnableParallel({
#         "context": retriever | RunnableLambda(_format_docs),
#         "question": RunnablePassthrough()
#     })
#     chain = parallel | prompt | llm | StrOutputParser()
#     query = "Extract property details (name, address, type, year built, sqft, units, amenities)"
#     out = chain.invoke(query)
#     # strip triple backticks or ```json fences
#     out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.I|re.M)
#     try:
#         import json
#         return json.loads(out)
#     except Exception:
#         return {"raw": out}

# # ------------- METRICS -------------

# def compute_metrics(inputs: Dict[str, Any]) -> Dict[str, Any]:
#     purchase_price = _to_number(inputs.get("purchase_price"))
#     noi = _to_number(inputs.get("net_operating_income"))
#     annual_debt_service = _to_number(inputs.get("annual_debt_service"))
#     cfb4tax = _to_number(inputs.get("cash_flow_before_taxes"))
#     equity_invested = _to_number(inputs.get("equity_invested"))
#     total_building_sqft = _to_number(inputs.get("total_building_sqft"))
#     total_units = _to_number(inputs.get("total_units"))
#     gpr = _to_number(inputs.get("gross_potential_rent"))
#     opex = _to_number(inputs.get("operating_expenses"))
#     current_rent_total = _to_number(inputs.get("current_rent_total"))
#     market_rent_total = _to_number(inputs.get("market_rent_total"))
#     rent_gap_pct = inputs.get("rent_gap_pct")
#     if rent_gap_pct is None and current_rent_total and market_rent_total:
#         rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
#     cap_rate = (noi / purchase_price) if (noi and purchase_price) else None
#     dscr = (noi / annual_debt_service) if (noi and annual_debt_service) else None
#     coc = (cfb4tax / equity_invested) if (cfb4tax and equity_invested) else None
#     price_per_sf = (purchase_price / total_building_sqft) if (purchase_price and total_building_sqft) else None
#     price_per_unit = (purchase_price / total_units) if (purchase_price and total_units) else None
#     break_even_occ = ((opex or 0.0) + (annual_debt_service or 0.0)) / gpr if gpr else None
#     return {
#         "cap_rate": cap_rate,
#         "dscr": dscr,
#         "coc_return": coc,
#         "irr_5yr": None,
#         "rent_gap_pct": rent_gap_pct,
#         "price_per_sqft": price_per_sf,
#         "price_per_unit": price_per_unit,
#         "break_even_occupancy": break_even_occ
#     }

# # ------------- ORCHESTRATION -------------

# @traceable(name="setup_rag")
# def setup_rag(pdf_path: str):
#     docs = load_pdf_multi(pdf_path)
#     splits = split_documents(docs)
#     return build_vectorstore(splits)

# def main(pdf_path: str = PDF_PATH):
#     print("\n--- STRUCTURED TABLE PARSING ---")
#     tables = extract_tables_to_dataframes(pdf_path)
#     # Save debug CSVs for structured tables found
#     for i, df in enumerate(tables["rent_roll"]):
#         df.to_csv(f"debug_structured_rent_roll_{i+1}.csv", index=False)
#     for i, df in enumerate(tables["t12"]):
#         df.to_csv(f"debug_structured_t12_{i+1}.csv", index=False)

#     rent_roll_summary = aggregate_rent_roll(tables["rent_roll"])
#     t12_summary = aggregate_t12(tables["t12"])
#     print("\nRent Roll Summary:", rent_roll_summary)
#     print("T12 Summary:", t12_summary)

#     print("\n--- RAG NARRATIVE EXTRACTION ---")
#     vs = setup_rag(pdf_path)
#     narrative = extract_narrative_fields(vs)
#     print("Narrative Fields:", narrative)

#     inputs = {
#         "net_operating_income": t12_summary.get("net_operating_income"),
#         "operating_expenses": t12_summary.get("operating_expenses"),
#         "gross_potential_rent": t12_summary.get("gross_potential_rent"),
#         "current_rent_total": rent_roll_summary.get("current_rent_total"),
#         "market_rent_total": rent_roll_summary.get("market_rent_total"),
#         "rent_gap_pct": rent_roll_summary.get("rent_gap_pct"),
#         "total_units": rent_roll_summary.get("total_units"),
#         "total_building_sqft": narrative.get("total_building_sqft"),
#         "purchase_price": None,
#         "annual_debt_service": None,
#         "cash_flow_before_taxes": None,
#         "equity_invested": None,
#     }

#     print("\n--- METRICS ---")
#     metrics = compute_metrics(inputs)
#     for k, v in metrics.items():
#         if isinstance(v, float):
#             if "rate" in k or "pct" in k or "occupancy" in k:
#                 print(f"{k}: {v:.2%}")
#             elif "price" in k:
#                 print(f"{k}: ${v:,.2f}")
#             else:
#                 print(f"{k}: {v:,.2f}")
#         else:
#             print(f"{k}: {v}")

#     print("\n--- QUICK SUMMARY ---")
#     print(f"Property: {narrative.get('property_name', 'Unknown')}")
#     print(f"Address: {narrative.get('property_address', 'Unknown')}")
#     print(f"Year Built: {narrative.get('year_built', 'Unknown')}")
#     print(f"SqFt: {narrative.get('total_building_sqft', 'Unknown')}")
#     print(f"NOI (from T12): {t12_summary.get('net_operating_income')}")
#     print(f"Expenses (from T12): {t12_summary.get('operating_expenses')}")
#     print(f"GPR (from T12): {t12_summary.get('gross_potential_rent')}")
#     print(f"Rent Gap % (from Rent Roll): {metrics.get('rent_gap_pct'):.2f}%" if metrics.get("rent_gap_pct") else "Rent Gap %: Unknown")

# if __name__ == "__main__":
#     main(PDF_PATH)
























# hybrid_underwriting.py
# -*- coding: utf-8 -*-
"""
Hybrid CRE Underwriting Pipeline: Structured Tables + RAG + Metrics
Supports multiple inputs: PDF / CSV / Excel / TXT / JSON
Requires:
pip install -U pandas numpy langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith pdfplumber openpyxl
"""
import os
import re
import math
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
import csv

# Avoid macOS OpenMP crash (FAISS/numpy)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import pdfplumber
from dotenv import load_dotenv

from langsmith import traceable
# PDF loaders (community)
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PyMuPDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# ------------- CONFIG -------------
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TABLE_CHUNK_SIZE = int(os.getenv("TABLE_CHUNK_SIZE", "1000"))
TABLE_CHUNK_OVERLAP = int(os.getenv("TABLE_CHUNK_OVERLAP", "150"))

# ------------- HELPERS -------------
def _to_number(x: Any) -> Optional[float]:
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
        "term", "start", "end", "sf", "sqft", "occupied", "vacant","Current Rent Total","Market Rent Total"
    ]
    score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
    return score >= 3

def _guess_is_t12(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    wanted = [
        "income", "revenue", "rent", "gpr", "noi", "expenses",
        "egi", "vacancy", "operating", "total", "Gross Potential Rent", "Effective Gross Income","Operating Expenses","Net Operating Income"
    ]
    score = sum(any(w in str(c).lower() for w in wanted) for c in cols)
    return score >= 2



def compute_irr(cash_flows: List[float], guess: float = 0.1) -> Optional[float]:
    """Compute IRR using Newton-Raphson method."""
    if not cash_flows:
        return None
    try:
        return np.irr(cash_flows) * 100  # in percentage
    except:
        return None


# ------------- FILE / DOCUMENT LOADING (multiple types) -------------

def _df_to_document(df: pd.DataFrame, meta: Dict[str, Any]) -> Document:
    """Serialize DataFrame to text document for RAG / splitting while keeping metadata."""
    # Convert header + rows to a text representation
    text = df.to_csv(index=False)
    return Document(page_content=text, metadata=meta)

@traceable(name="load_pdf_multi")
def load_pdf_multi(path: str) -> List[Document]:
    """Attempt multiple PDF loaders and return unique documents"""
    docs: List[Document] = []
    loaders = [
        ("PDFPlumberLoader", PDFPlumberLoader(path)),
        ("PyPDFLoader", PyPDFLoader(path)),
        ("UnstructuredPDFLoader (OCR)", UnstructuredPDFLoader(path, strategy="ocr_only")),
        ("PyPDFium2Loader", PyPDFium2Loader(path)),
        ("PyMuPDFLoader", PyMuPDFLoader(path)),
        ("PDFMinerLoader", PDFMinerLoader(path)),
        ("PDFMinerPDFasHTMLLoader", PDFMinerPDFasHTMLLoader(path)),
    ]
    for name, loader in loaders:
        try:
            new_docs = loader.load()
            docs.extend(new_docs)
            print(f"{name} extracted {len(new_docs)} docs from {os.path.basename(path)}")
        except Exception as e:
            print(f"{name} failed for {os.path.basename(path)}: {e}")
    # dedupe by content
    seen = set()
    unique_docs: List[Document] = []
    for d in docs:
        key = (d.page_content or "").strip()
        if key and key not in seen:
            unique_docs.append(d)
            seen.add(key)
    print(f"Total unique pdf docs after merging for {os.path.basename(path)}: {len(unique_docs)}")
    return unique_docs

def load_csv_as_documents(path: str) -> List[Document]:
    """Load CSV into a single Document (CSV text) and a structured DataFrame in metadata"""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read CSV {path}: {e}")
        return [Document(page_content="", metadata={"source": path})]
    doc = _df_to_document(df, {"source": path, "type": "csv", "rows": len(df)})
    # attach DataFrame as metadata for downstream structured parsing (we'll still store debug CSVs)
    doc.metadata["dataframe"] = df
    return [doc]

def load_excel_as_documents(path: str) -> List[Document]:
    """Load all sheets from Excel as separate Documents"""
    docs = []
    try:
        xls = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"Failed to read Excel {path}: {e}")
        return [Document(page_content="", metadata={"source": path})]
    for sheet_name, df in xls.items():
        meta = {"source": path, "sheet": sheet_name, "type": "excel", "rows": len(df)}
        doc = _df_to_document(df, meta)
        doc.metadata["dataframe"] = df
        docs.append(doc)
    return docs

def load_text_or_json(path: str) -> List[Document]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read()
    except Exception as e:
        print(f"Failed to read text/json {path}: {e}")
        txt = ""
    return [Document(page_content=txt, metadata={"source": path, "type": "text_or_json"})]

def load_files(paths: List[str]) -> List[Document]:
    """Top-level file loader that supports PDF, CSV, Excel, TXT/JSON"""
    all_docs: List[Document] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Path not found: {p}; skipping.")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in [".pdf"]:
            all_docs.extend(load_pdf_multi(p))
        elif ext in [".csv"]:
            all_docs.extend(load_csv_as_documents(p))
        elif ext in [".xls", ".xlsx"]:
            all_docs.extend(load_excel_as_documents(p))
        elif ext in [".txt", ".json"]:
            all_docs.extend(load_text_or_json(p))
        else:
            # fallback: attempt to read as text; also try pandas for unknown tabular files
            try:
                df = pd.read_csv(p)
                doc = _df_to_document(df, {"source": p, "type": "csv_fallback", "rows": len(df)})
                doc.metadata["dataframe"] = df
                all_docs.append(doc)
            except Exception:
                all_docs.extend(load_text_or_json(p))
    print(f"Loaded a total of {len(all_docs)} documents from inputs.")
    return all_docs

# ------------- TABLE PARSING (pdfplumber) -------------
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

# ------------- TEXT-BASED FALLBACK PARSERS -------------
def _extract_amounts_from_line(line: str) -> List[float]:
    """Return list of monetary numbers found in a line."""
    amounts = []
    # A slightly more robust regex for money/ints
    for m in re.finditer(r"(-?\$?\d{1,3}(?:[,\d]{0,}|(?:\d+))(?:\.\d{1,2})?)", line):
        val = m.group(0)
        if '$' in val or re.search(r"\d", val):
            num = _to_number(val)
            if num is not None:
                amounts.append(num)
    return amounts

@traceable(name="parse_rent_roll_from_text")
def parse_rent_roll_from_text(path: str) -> pd.DataFrame:
    """
    Heuristic parser: scan page text for suites/tenant lines and nearby numbers.
    Returns a dataframe with candidate rows and parsed fields (best-effort).
    """
    candidates = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                for idx, line in enumerate(lines):
                    if re.search(r"(^suite\s*\d+)|(^\d{2,4}\b)", line.lower()) or re.search(r"\b(suite|ste|#)\b", line.lower()):
                        window = " | ".join(lines[max(0, idx-2): idx+4])
                        amounts = _extract_amounts_from_line(window)
                        sqft = None
                        m_sq = re.search(r"(\d{3,6})(?:\s?(?:sf|sq\.?ft|s\.f\.))", window.replace(",", "").lower())
                        if m_sq:
                            sqft = _to_number(m_sq.group(1))
                        lease = None
                        m_lease = re.search(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\w\.-]*\s*\-?\s*\d{4})", window, flags=re.I)
                        if m_lease:
                            lease = m_lease.group(0)
                        m_tenant = re.search(r"\b(?:suite|ste|#)?\s*\d{0,4}\s*[-:]*\s*([A-Z][A-Za-z0-9&,\.\- ]{2,80})", window)
                        tenant = m_tenant.group(1).strip() if m_tenant else None
                        candidates.append({
                            "page": i+1,
                            "window": window,
                            "tenant": tenant,
                            "sqft": sqft,
                            "amounts": amounts,
                            "lease_hint": lease
                        })
    except Exception as e:
        print(f"parse_rent_roll_from_text failed for {path}: {e}")
    df = pd.DataFrame(candidates)
    def pick_amount(row):
        if not row.get("amounts"):
            return None
        nums = sorted(row["amounts"])
        return nums[-1]
    if not df.empty:
        df["best_amount"] = df.apply(pick_amount, axis=1)
    df.to_csv("debug_rent_roll_candidates.csv", index=False)
    return df

@traceable(name="parse_t12_from_text")
def parse_t12_from_text(path: str) -> Dict[str, Optional[float]]:
    text_all = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_all += "\n" + (page.extract_text() or "")
    except Exception as e:
        print(f"parse_t12_from_text failed for {path}: {e}")
        text_all = ""

    txt = re.sub(r"\s{2,}", " ", text_all)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    candidates = []
    for ln in lines:
        if re.search(r"(gross.*potential.*rent|potential.*rent|gpr|effective gross|effective gross income|net operating income|operating expenses|total expenses|vacancy|credit loss|other income|misc income|parking|laundry)", ln, flags=re.I):
            nums = _extract_amounts_from_line(ln)
            candidates.append({"line": ln, "amounts": nums})
    with open("debug_t12_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["line", "amounts"])
        writer.writeheader()
        for c in candidates:
            writer.writerow({"line": c["line"], "amounts": ";".join(str(a) for a in c["amounts"])})
    def pick_for_pattern(pats):
        for c in candidates:
            for p in pats:
                if re.search(p, c["line"], flags=re.I):
                    if c["amounts"]:
                        return c["amounts"][-1]
        return None
    gpr = pick_for_pattern([r"gross.*potential.*rent", r"potential.*rent", r"gpr"])
    vacancy = pick_for_pattern([r"vacancy", r"credit.*loss", r"loss.*to.*lease"])
    other_income = pick_for_pattern([r"other.*income", r"misc.*income", r"parking", r"laundry", r"storage"])
    egi_candidate = pick_for_pattern([r"effective gross income", r"effective.*gross", r"egi"])
    opex = pick_for_pattern([r"operating expenses", r"total expenses", r"expenses", r"total operating expenses"])
    noi = pick_for_pattern([r"net operating income", r"noi"])
    egi = egi_candidate
    if egi is None and gpr is not None:
        egi = gpr - (vacancy or 0.0) + (other_income or 0.0)
    return {
        "gross_potential_rent": gpr,
        "vacancy": vacancy,
        "other_income": other_income,
        "effective_gross_income": egi,
        "operating_expenses": opex,
        "net_operating_income": noi
    }

# ------------- AGGREGATION FROM STRUCTURED TABLES -------------
# @traceable(name="aggregate_rent_roll")
# def aggregate_rent_roll(dfs: List[pd.DataFrame], fallback_pdf_paths: List[str] = []) -> Dict[str, Any]:
#     """
#     Compute current_rent_total, market_rent_total and rent_gap_pct from structured dfs.
#     If none found, optionally attempt text-based parsing on provided fallback_pdf_paths.
#     """
#     if dfs:
#         total_units = 0
#         current_rent_total = 0.0
#         market_rent_total = 0.0
#         for df in dfs:
#             df = _clean_dataframe(df)
#             cols = list(df.columns)
#             col_rent = _first_match(cols, [r"(rent|contract|scheduled|actual)", r"(^|[^a-z])rent([^a-z]|$)", r"Current.*Rent.*Total", r"base.*rent", r"annual.*rent", r"monthly.*rent"])
#             col_market = _first_match(cols, [r"(market|asking|proforma).*rent",r"Market.*Rent.*Total", r"asking.*rent"])
#             col_psf = _first_match(cols, [r"psf", r"per\s*sf"])
#             col_sf = _first_match(cols, [r"sf", r"sq.?ft", r"area"])
#             total_units += len(df)
#             cur_sum = sum(_to_number(v) or 0.0 for v in df[col_rent].values) if col_rent else 0.0
#             mkt_sum = sum(_to_number(v) or 0.0 for v in df[col_market].values) if col_market else 0.0
#             if (mkt_sum == 0.0) and col_psf and col_sf:
#                 monthly = bool(re.search(r"(\/mo|per\s*month|monthly)", col_psf.lower()))
#                 for _, row in df.iterrows():
#                     psf = _to_number(row.get(col_psf))
#                     sf = _to_number(row.get(col_sf))
#                     if psf and sf:
#                         mkt_sum += psf * sf * (12 if monthly else 1)
#             current_rent_total += cur_sum
#             market_rent_total += mkt_sum
#         rent_gap_pct = None
#         if market_rent_total > 0:
#             rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
#         return {
#             "total_units": total_units,
#             "current_rent_total": current_rent_total if current_rent_total > 0 else None,
#             "market_rent_total": market_rent_total if market_rent_total > 0 else None,
#             "rent_gap_pct": rent_gap_pct,
#         }
#     # fallback: try parse from provided PDFs using text heuristics
#     for p in fallback_pdf_paths:
#         try:
#             text_df = parse_rent_roll_from_text(p)
#             if not text_df.empty:
#                 cand = text_df.copy()
#                 cand["best_amount"] = cand["best_amount"].apply(lambda x: x if x and x > 1000 else None)
#                 total_units = len(cand)
#                 current_rent_total = cand["best_amount"].dropna().sum() if "best_amount" in cand else None
#                 return {
#                     "total_units": total_units if total_units>0 else None,
#                     "current_rent_total": current_rent_total if current_rent_total and current_rent_total>0 else None,
#                     "market_rent_total": None,
#                     "rent_gap_pct": None,
#                 }
#         except Exception:
#             continue
#     return {"total_units": None, "current_rent_total": None, "market_rent_total": None, "rent_gap_pct": None}


def aggregate_rent_roll(dfs: List[pd.DataFrame], fallback_pdf_paths: List[str] = []) -> Dict[str, Any]:
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

        # Ensure rent gap is always calculated if possible
        rent_gap_pct = None
        if market_rent_total > 0 and current_rent_total is not None:
            rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0
        elif current_rent_total and market_rent_total == 0:
            rent_gap_pct = 0.0  # fallback if market rent missing

        return {
            "total_units": total_units or None,
            "current_rent_total": current_rent_total or None,
            "market_rent_total": market_rent_total or None,
            "rent_gap_pct": rent_gap_pct,
        }
    # fallback: text-based parsing
    for p in fallback_pdf_paths:
        try:
            text_df = parse_rent_roll_from_text(p)
            if not text_df.empty:
                cand = text_df.copy()
                cand["best_amount"] = cand["best_amount"].apply(lambda x: x if x and x > 1000 else None)
                total_units = len(cand)
                current_rent_total = cand["best_amount"].dropna().sum() if "best_amount" in cand else None
                # rent gap can't be computed without market rent
                rent_gap_pct = None
                return {
                    "total_units": total_units if total_units>0 else None,
                    "current_rent_total": current_rent_total if current_rent_total and current_rent_total>0 else None,
                    "market_rent_total": None,
                    "rent_gap_pct": rent_gap_pct,
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

# # ------------- RAG & Narrative -------------
def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="split_documents")
def split_documents(docs, chunk_size=TABLE_CHUNK_SIZE, chunk_overlap=TABLE_CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_splits = []
    for doc in docs:
        text = doc.page_content or ""
        # rough table detection: presence of commas+newlines or many numeric tokens
        if re.search(r"\d+\s+\d+", text) and ("\n" in text or "\t" in text):
            final_splits.append(doc)
        else:
            final_splits.extend(splitter.split_documents([doc]))
    print(f"Total chunks after table-aware splitting: {len(final_splits)}")
    return final_splits

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(splits, emb)

@traceable(name="extract_narrative_fields")
def extract_narrative_fields(vs) -> Dict[str, Any]:
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract ONLY from the provided context. If not found, write 'I don't know'. Respond in strict JSON."),
        ("human", """Extract these fields:
- property_name
- property_address
- property_type
- year_built
- renovation_year
- number_of_stories
- total_units_or_suites
- total_building_sqft
- amenities

Context:
{context}""")
    ])
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(_format_docs),
        "question": RunnablePassthrough()
    })
    chain = parallel | prompt | llm | StrOutputParser()
    query = "Extract property details (name, address, type, year built, sqft, units, amenities)"
    out = chain.invoke(query)
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.I|re.M)
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


# def _format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=TABLE_CHUNK_SIZE, chunk_overlap=TABLE_CHUNK_OVERLAP):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     final_splits = []
#     for doc in docs:
#         final_splits.extend(splitter.split_documents([doc]))
#     print(f"Total chunks after splitting: {len(final_splits)}")
#     return final_splits

# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = OpenAIEmbeddings(model=EMBED_MODEL)
#     return FAISS.from_documents(splits, emb)

# @traceable(name="extract_all_fields")
# def extract_all_fields(vs) -> Dict[str, Any]:
#     """Extract ALL required underwriting fields from vector DB."""
#     retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Extract ONLY from the provided context. If not found, write 'null'. Respond strictly in JSON."),
#         ("human", """Extract these fields:
# - property_name
# - property_address
# - property_type
# - year_built
# - renovation_year
# - number_of_stories
# - total_units_or_suites
# - total_building_sqft
# - amenities

# Financials:
# - gross_potential_rent
# - vacancy
# - effective_gross_income
# - operating_expenses
# - net_operating_income
# - purchase_price
# - annual_debt_service
# - equity_invested
# - current_rent_total
# - market_rent_total

# Context:
# {context}""")
#     ])

#     chain = (retriever | RunnableLambda(_format_docs)) | prompt | llm | StrOutputParser()
#     query = "Extract all property and financial fields"
#     out = chain.invoke(query)
#     out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.I|re.M)
#     try:
#         return json.loads(out)
#     except Exception:
#         return {"raw": out}

# ------------- METRICS -------------
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
        purchase_price = 5000000 # assume 7.5% cap if price not given

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


# def compute_metrics(extracted: Dict[str, Any]) -> Dict[str, Any]:
#     """Compute metrics directly from extracted JSON fields."""
#     noi = _to_number(extracted.get("net_operating_income"))
#     gpr = _to_number(extracted.get("gross_potential_rent"))
#     egi = _to_number(extracted.get("effective_gross_income"))
#     opex = _to_number(extracted.get("operating_expenses"))

#     current_rent_total = _to_number(extracted.get("current_rent_total"))
#     market_rent_total = _to_number(extracted.get("market_rent_total"))
#     rent_gap_pct = None
#     if current_rent_total and market_rent_total:
#         rent_gap_pct = (market_rent_total - current_rent_total) / market_rent_total * 100.0

#     sqft = _to_number(extracted.get("total_building_sqft"))
#     units = _to_number(extracted.get("total_units_or_suites"))

#     purchase_price = _to_number(extracted.get("purchase_price"))
#     if not purchase_price and noi:
#         purchase_price = noi / 0.075  # assume 7.5% cap if price not given

#     annual_debt_service = _to_number(extracted.get("annual_debt_service")) or (purchase_price * 0.05 if purchase_price else None)
#     equity_invested = _to_number(extracted.get("equity_invested")) or (purchase_price * 0.25 if purchase_price else None)

#     # Metrics
#     cap_rate = (noi / purchase_price) * 100 if (noi and purchase_price) else None
#     dscr = (noi / annual_debt_service) if (noi and annual_debt_service) else None
#     coc = ((noi - annual_debt_service) / equity_invested) * 100 if (noi and annual_debt_service and equity_invested) else None
#     price_per_sf = (purchase_price / sqft) if (purchase_price and sqft) else None
#     price_per_unit = (purchase_price / units) if (purchase_price and units) else None
#     break_even_occ = ((opex or 0.0) + (annual_debt_service or 0.0)) / egi * 100 if egi else None

#     # IRR (5yr, using NOI - debt service as proxy cash flow)
#     irr_5yr = None
#     if noi and annual_debt_service and equity_invested:
#         base_cash_flow = noi - annual_debt_service
#         cash_flows = [-equity_invested] + [base_cash_flow * ((1 + 0.02) ** i) for i in range(1, 6)]
#         try:
#             irr_5yr = compute_irr(cash_flows)
#         except Exception:
#             irr_5yr = None

#     return {
#         "cap_rate": cap_rate,
#         "dscr": dscr,
#         "coc_return": coc,
#         "irr_5yr": irr_5yr,
#         "rent_gap_pct": rent_gap_pct,
#         "price_per_sqft": price_per_sf,
#         "price_per_unit": price_per_unit,
#         "break_even_occupancy": break_even_occ,
#     }


# ------------- AI SUMMARY -------------
# def generate_underwriting_summary(narrative: Dict[str, Any], metrics: Dict[str, Any]) -> str:
#     """Use OpenAI to generate a brief professional underwriting summary."""
#     prompt = f"""
# You are a real estate underwriting analyst. 
# Given the following property narrative and financial metrics, create a concise, professional underwriting summary:

# Narrative:
# {json.dumps(narrative, indent=2)}

# Metrics:
# {json.dumps(metrics, indent=2)}

# Summary:
# """
#     try:
#         response = ChatOpenAI(
#             model="gpt-4",
#             temperature=0.3,
#         )
#         messages=[{"role": "user", "content": prompt}]
#         response = response.invoke(messages)
#         summary = response.content.strip()
#         return summary
#     except Exception as e:
#         return f"AI summary generation failed: {e}"


def generate_underwriting_summary(narrative: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Generate a professional, actionable underwriting summary.
    Includes property overview, key metrics, rent gap %, and recommendation hint.
    """
    prompt = f"""
You are a senior real estate underwriting analyst. 
Given the property narrative and financial metrics below, create a concise, professional underwriting summary.
The summary should allow an investor to quickly assess whether to consider buying or not. 
Include key highlights, property overview, financials (NOI, GPR, rent gap %, IRR), and any important observations.

Property Narrative:
{json.dumps(narrative, indent=2)}

Financial Metrics:
{json.dumps(metrics, indent=2)}

Summary:
"""
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        summary = response.content.strip()

        # Optional: Add fallback for empty or invalid content
        if not summary:
            summary = "AI summary generation returned empty. Check inputs or model."
        return summary
    except Exception as e:
        return f"AI summary generation failed: {e}"

# ------------- AI ANALYSIS -------------
def generate_underwriting_analysis(narrative: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Use OpenAI to generate a professional underwriting analysis including:
    1. Investment Recommendation
    2. Key Investment Highlights
    3. Risk Considerations
    """
    prompt = f"""
You are a real estate underwriting analyst. 

Given the property narrative and financial metrics below, provide a structured professional underwriting analysis including:

1. Investment Recommendation (PASS, CONSIDER, BUY)
2. Key Investment Highlights (bullet points)
3. Risk Considerations (bullet points)

Narrative:
{json.dumps(narrative, indent=2)}

Metrics:
{json.dumps(metrics, indent=2)}

Please respond in JSON format like:
{{
  "investment_recommendation": "...",
  "key_investment_highlights": ["...","..."],
  "risk_considerations": ["...","..."]
}}
"""
    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0.3)
        response = chat.invoke([{"role": "user", "content": prompt}])
        # Ensure JSON parsing
        analysis = json.loads(response.content)
        return analysis
    except Exception as e:
        return {
            "investment_recommendation": f"AI generation failed: {e}",
            "key_investment_highlights": [],
            "risk_considerations": []
        }

# ------------- ORCHESTRATION -------------
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid CRE Underwriting Pipeline - accept multiple files")
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input files (pdf/csv/xlsx/txt/json). Multiple allowed.")
    parser.add_argument("--overrides", "-o", help="Optional JSON string with manual overrides (purchase_price, annual_debt_service, etc.)")
    args = parser.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else {}
    run_pipeline(args.inputs, overrides=overrides)



# def run_pipeline(inputs: List[str]):
#     print("\n--- LOADING FILES ---")
#     docs = load_files(inputs)

#     print("\n--- BUILDING VECTOR DB ---")
#     splits = split_documents(docs)
#     vs = build_vectorstore(splits)

#     print("\n--- EXTRACTING FIELDS ---")
#     extracted = extract_all_fields(vs)
#     print(json.dumps(extracted, indent=2))

#     print("\n--- METRICS ---")
#     metrics = compute_metrics(extracted)
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     print("\n--- AI UNDERWRITING ANALYSIS ---")
#     ai_analysis = generate_underwriting_analysis(extracted, metrics)
#     print(json.dumps(ai_analysis, indent=2))

#     print("\n--- QUICK SUMMARY ---")
#     print(f"Property: {extracted.get('property_name')}")
#     print(f"Address: {extracted.get('property_address')}")
#     print(f"Year Built: {extracted.get('year_built')}")
#     print(f"SqFt: {extracted.get('total_building_sqft')}")
#     print(f"NOI: {extracted.get('net_operating_income')}")
#     print(f"GPR: {extracted.get('gross_potential_rent')}")
#     print(f"Rent Gap %: {metrics.get('rent_gap_pct')}")
#     print(f"5-Year IRR: {metrics.get('irr_5yr')}%")
#     print(f"Investment Recommendation: {ai_analysis.get('investment_recommendation')}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="CRE Underwriting Pipeline (VectorDB driven)")
#     parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input files (pdf/csv/xlsx/txt/json)")
#     args = parser.parse_args()
#     run_pipeline(args.inputs)





#     utils/
# ├── __init__.py
# ├── file_loaders.py
# ├── table_parsers.py
# ├── text_parsers.py
# ├── aggregation.py
# ├── metrics.py
# ├── rag_narrative.py
# ├── ai_summary.py
# ├── ai_analysis.py
# ├── helpers.py
