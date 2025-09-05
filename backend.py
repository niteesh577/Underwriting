# app.py
import os
import json
import tempfile
import shutil
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() 


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_number(value):
    """Convert strings like '$25,000' or '25,000' to float"""
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "")
    try:
        return float(value)
    except:
        return None
    

def save_to_supabase(narrative_fields, metrics, t12_summary, ai_summary, ai_analysis):
    data = {
        "property_name": narrative_fields.get("property_name"),
        "address": narrative_fields.get("property_address"),
        "year_built": narrative_fields.get("year_built"),
        "sqft": clean_number(narrative_fields.get("total_building_sqft")),
        "metrics": {
            "cap_rate": clean_number(metrics.get("cap_rate")),
            "dscr": clean_number(metrics.get("dscr")),
            "coc_return": clean_number(metrics.get("coc_return")),
            "irr_5yr": clean_number(metrics.get("irr_5yr")),
            "rent_gap_pct": clean_number(metrics.get("rent_gap_pct")),
            "price_per_sqft": clean_number(metrics.get("price_per_sqft")),
            "price_per_unit": clean_number(metrics.get("price_per_unit")),
            "break_even_occupancy": clean_number(metrics.get("break_even_occupancy")),
        },
        "t12_summary": t12_summary,
        "ai_summary": ai_summary,
        "ai_analysis": ai_analysis,
        "created_at": datetime.utcnow().isoformat()
    }
    try:
        supabase.table("Underwriting").insert(data).execute()
    except Exception as e:
        print("Supabase insert failed:", e)



# Import your existing pipeline functions
from tester import (
    load_files,
    extract_tables_to_dataframes_from_docs,
    aggregate_rent_roll,
    aggregate_t12,
    split_documents,
    extract_narrative_fields,
    compute_metrics,
    generate_underwriting_summary,
    generate_underwriting_analysis  # <-- Added
)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="CRE Underwriting API", version="1.0")

# Allow frontend origin
origins = [
    "http://localhost:5173",  # Vite/React dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicitly use OpenAI embeddings for FAISS vectorstore
def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    return FAISS.from_documents(docs, embeddings)

@app.post("/underwrite")
async def underwrite(
    files: List[UploadFile] = File(...),
    overrides: str = Form(default="{}"),
):
    """
    Upload one or more files (PDF, Excel, CSV, JSON, TXT) and get underwriting metrics.
    Optionally pass overrides (JSON string) to inject purchase price, debt service, etc.
    """
    tmpdir = tempfile.mkdtemp()
    paths = []

    try:
        # Save uploads to temp files
        for f in files:
            path = os.path.join(tmpdir, f.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            paths.append(path)

        # Load and process files
        docs = load_files(paths)
        table_dfs = extract_tables_to_dataframes_from_docs(docs)

        rent_roll_summary = aggregate_rent_roll(table_dfs.get("rent_roll", []), paths)
        t12_summary = aggregate_t12(table_dfs.get("t12", []), paths)

        splits = split_documents(docs)
        vs = build_vectorstore(splits)
        narrative_fields = extract_narrative_fields(vs)

        # Parse overrides
        try:
            overrides_dict: Dict[str, Any] = json.loads(overrides)
        except Exception:
            overrides_dict = {}

        # Compute metrics
        metrics = compute_metrics(
            t12_summary,
            rent_roll_summary,
            narrative_fields,
            overrides=overrides_dict
        )

        # Generate AI underwriting summary
        ai_summary = generate_underwriting_summary(narrative_fields, metrics)

        ai_analysis = generate_underwriting_analysis(narrative_fields, metrics)

        # Save to Supabase
        save_to_supabase(narrative_fields, metrics, t12_summary, ai_summary, ai_analysis)

        result = {
    "rent_roll_summary": rent_roll_summary,
    "t12_summary": t12_summary,
    "narrative_fields": narrative_fields,
    "metrics": metrics,
    "ai_summary": ai_summary,  # <-- includes recommendation, highlights, risks
    "quick_summary": {
        "property": narrative_fields.get("property_name"),
        "address": narrative_fields.get("property_address"),
        "year_built": narrative_fields.get("year_built"),
        "sqft": narrative_fields.get("total_building_sqft"),
        "NOI": t12_summary.get("net_operating_income"),
        "Expenses": t12_summary.get("operating_expenses"),
        "GPR": t12_summary.get("gross_potential_rent"),
        "Rent Gap %": metrics.get("rent_gap_pct"),
        "Cap Rate": metrics.get("cap_rate"),
        "DSCR": metrics.get("dscr"),
        "CoC Return": metrics.get("coc_return"),
        "Price per SqFt": metrics.get("price_per_sqft"),
        "Price per Unit": metrics.get("price_per_unit"),
        "Break-even Occupancy": metrics.get("break_even_occupancy"),
        # Include AI analysis directly for frontend convenience
        "Investment Recommendation": ai_analysis.get("investment_recommendation"),
        "Key Investment Highlights": ai_analysis.get("key_investment_highlights"),
        "Risk Considerations": ai_analysis.get("risk_considerations"),
    },
}

        return JSONResponse(content=result)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
