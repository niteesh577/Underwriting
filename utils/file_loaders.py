from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PyMuPDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
)

import pandas as pd
import numpy as np
import pdfplumber
from dotenv import load_dotenv

from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import csv
import os


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

