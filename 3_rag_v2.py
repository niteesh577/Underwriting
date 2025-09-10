# # pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

# import os
# from dotenv import load_dotenv

# from langsmith import traceable  # <-- key import

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain.schema import Document
# # --- LangSmith env (make sure these are set) ---
# # LANGCHAIN_TRACING_V2=true
# # LANGCHAIN_API_KEY=...
# # LANGCHAIN_PROJECT=pdf_rag_demo

# load_dotenv()

# PDF_PATH = "ParkCenter_400_OM.pdf"  # change to your file

# # ---------- traced setup steps ----------
# # @traceable(name="load_pdf")
# # def load_pdf(path: str):
# #     loader = PyPDFLoader(path)
# #     return loader.load()  # list[Document]

# @traceable(name="load_pdf")
# def load_pdf(path: str) -> list[Document]:
#     """
#     Load PDF using PDFPlumber for better table + text extraction.
#     Falls back to PyPDFLoader if PDFPlumber fails.
#     """
#     try:
#         loader = PDFPlumberLoader(path)
#         docs = loader.load()
#     except Exception as e:
#         print(f"PDFPlumber failed: {e}, falling back to PyPDFLoader...")
#         from langchain.document_loaders import PyPDFLoader
#         loader = PyPDFLoader(path)
#         docs = loader.load()
    
#     return docs

# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=1000, chunk_overlap=150):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     return splitter.split_documents(docs)

# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = OpenAIEmbeddings(model="text-embedding-3-small")
#     # FAISS.from_documents internally calls the embedding model:
#     vs = FAISS.from_documents(splits, emb)
#     return vs

# # You can also trace a “setup” umbrella span if you want:
# @traceable(name="setup_pipeline")
# def setup_pipeline(pdf_path: str):
#     docs = load_pdf(pdf_path)
#     splits = split_documents(docs)
#     vs = build_vectorstore(splits)
#     return vs

# # ---------- pipeline ----------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
#     ("human", "Question: {question}\n\nContext:\n{context}")
# ])

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# # Build the index under traced setup
# vectorstore = setup_pipeline(PDF_PATH)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# parallel = RunnableParallel({
#     "context": retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough(),
# })

# chain = parallel | prompt | llm | StrOutputParser()

# # ---------- run a query (also traced) ----------

# print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

# config = {
#     "run_name": "pdf_rag_query"
# }

# while True:
#     q = input("\nQ: ").strip()
#     ans = chain.invoke(q,config=config)
#     print("\nA:", ans)

# # Give the visible run name + tags/metadata so it’s easy to find:
# # config = {
# #     "run_name": "pdf_rag_query"
# # }

# # ans = chain.invoke(q, config=config)
# # print("\nA:", ans)











# # pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith pdfplumber

# import os
# from dotenv import load_dotenv

# from langsmith import traceable

# from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader, UnstructuredPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document

# # --- Load environment variables ---
# # Make sure you have in your .env:
# # OPENAI_API_KEY=...
# # LANGCHAIN_TRACING_V2=true
# # LANGCHAIN_API_KEY=...
# # LANGCHAIN_PROJECT=pdf_rag_demo
# load_dotenv()

# PDF_PATH = "ParkCenter_400_OM.pdf"  # <-- your file


# # -----------------------------
# # Load PDF with best parser
# # -----------------------------
# @traceable(name="load_pdf")
# def load_pdf(path: str) -> list[Document]:
#     """
#     Load PDF with priority:
#     1. PDFPlumberLoader (best for tables)
#     2. PyPDFLoader (text fallback)
#     3. OCR with UnstructuredPDFLoader (for scanned docs)
#     """
#     try:
#         loader = PDFPlumberLoader(path)
#         docs = loader.load()
#     except Exception as e1:
#         print(f"PDFPlumber failed: {e1}, trying PyPDFLoader...")
#         try:
#             loader = PyPDFLoader(path)
#             docs = loader.load()
#         except Exception as e2:
#             print(f"PyPDFLoader failed: {e2}, trying OCR...")
#             loader = UnstructuredPDFLoader(path, strategy="ocr_only")
#             docs = loader.load()
#     return docs


# # -----------------------------
# # Split into chunks
# # -----------------------------
# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=1000, chunk_overlap=150):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return splitter.split_documents(docs)


# # -----------------------------
# # Build Vectorstore
# # -----------------------------
# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = OpenAIEmbeddings(model="text-embedding-3-small")
#     vs = FAISS.from_documents(splits, emb)
#     return vs


# # -----------------------------
# # Setup pipeline
# # -----------------------------
# @traceable(name="setup_pipeline")
# def setup_pipeline(pdf_path: str):
#     docs = load_pdf(pdf_path)
#     splits = split_documents(docs)
#     vs = build_vectorstore(splits)
#     return vs


# # -----------------------------
# # RAG chain
# # -----------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer ONLY from the provided context. If not found, say 'I don't know'."),
#     ("human", "Question: {question}\n\nContext:\n{context}")
# ])

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# # Build vectorstore
# vectorstore = setup_pipeline(PDF_PATH)
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 4}
# )

# parallel = RunnableParallel({
#     "context": retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough(),
# })

# chain = parallel | prompt | llm | StrOutputParser()


# # -----------------------------
# # Interactive Q&A loop
# # -----------------------------
# print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

# config = {
#     "run_name": "pdf_rag_query"
# }

# while True:
#     q = input("\nQ: ").strip()
#     if not q:
#         continue
#     ans = chain.invoke(q, config=config)
#     print("\nA:", ans)






















# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith pdfplumber

import os
from dotenv import load_dotenv
import re

from langsmith import traceable
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


# --- Load env ---
load_dotenv()
PDF_PATH = "ParkCenter_400_OM.pdf"

#-----------------------------
# Multi-loader function
# -----------------------------
@traceable(name="load_pdf")
def load_pdf(path: str) -> list[Document]:
    """
    Load PDF with multiple loaders and merge results.
    Priority:
    - PDFPlumberLoader (best for tables)
    - PyPDFLoader (structured text)
    - UnstructuredPDFLoader OCR (scanned docs)
    """
    docs = []

    try:
        plumber_docs = PDFPlumberLoader(path).load()
        docs.extend(plumber_docs)
        print(f"PDFPlumber extracted {len(plumber_docs)} docs")
    except Exception as e1:
        print(f"PDFPlumber failed: {e1}")

    try:
        pypdf_docs = PyPDFLoader(path).load()
        docs.extend(pypdf_docs)
        print(f"PyPDFLoader extracted {len(pypdf_docs)} docs")
    except Exception as e2:
        print(f"PyPDFLoader failed: {e2}")

    try:
        ocr_docs = UnstructuredPDFLoader(path, mode="elements").load()
        docs.extend(ocr_docs)
        print(f"OCR extracted {len(ocr_docs)} docs")
    except Exception as e3:
        print(f"OCR failed: {e3}")

    # Deduplicate by page_content
    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    print(f"Total unique docs after merging: {len(unique_docs)}")
    return unique_docs





# @traceable(name="load_pdf")
# def load_pdf(path: str) -> list[Document]:
#     """
#     Try multiple PDF loaders and merge unique results.
#     """
#     docs = []
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

#     # Deduplicate by content
#     seen = set()
#     unique_docs = []
#     for d in docs:
#         if d.page_content not in seen:
#             unique_docs.append(d)
#             seen.add(d.page_content)

#     print(f"Total unique docs after merging: {len(unique_docs)}")
#     return unique_docs



# -----------------------------
# Split into chunks
# -----------------------------
# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=1000, chunk_overlap=150):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     return splitter.split_documents(docs)



@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    """
    Table-aware splitter:
    - Keeps full tables intact
    - Splits normal text with RecursiveCharacterTextSplitter
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_splits = []

    for doc in docs:
        text = doc.page_content

        # Detect "table-like" content: multiple numbers & spacing
        if re.search(r"\d+\s+\d+", text) and ("\n" in text or "\t" in text):
            # Treat the entire page/table as one chunk
            final_splits.append(doc)
        else:
            # Normal splitting
            final_splits.extend(splitter.split_documents([doc]))

    print(f"Total chunks after table-aware splitting: {len(final_splits)}")
    return final_splits


# -----------------------------
# Build Vectorstore
# -----------------------------
@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(splits, emb)
    return vs


# -----------------------------
# Setup pipeline
# -----------------------------
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


# -----------------------------
# RAG chain
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say 'I don't know'."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()


# -----------------------------
# Interactive Q&A loop
# -----------------------------
print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

config = {"run_name": "pdf_rag_query"}

while True:
    q = input("\nQ: ").strip()
    if not q:
        continue
    ans = chain.invoke(q, config=config)
    print("\nA:", ans)























