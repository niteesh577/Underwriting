
from typing import List, Dict, Any
from langsmith import traceable
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
import json


load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TABLE_CHUNK_SIZE = int(os.getenv("TABLE_CHUNK_SIZE", "1000"))
TABLE_CHUNK_OVERLAP = int(os.getenv("TABLE_CHUNK_OVERLAP", "150"))


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